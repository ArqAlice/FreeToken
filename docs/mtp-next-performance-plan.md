# 他ランタイムの実装に基づくMTP性能改善計画

調査日: 2026-09-09 UTC。対象: `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`、
RTX 5090 32 GB、CPU expert offload、OpenWebUIの新規チャット。
調査後の実施状況: 2026-09-10 JST に自動選択と追加最適化、実モデル・OpenWebUI検証まで実施。
[最新の実装・実測結果](../benchmarks/results/mtp-reuse/implementation.md)を参照。

- サイクル別の時間、採用長、expert cache miss と推定転送量の診断を追加。
  診断は通常の速度測定と別実行。行間のexpert重複率を直接収集する機能は未実装。
- 不要なGDN snapshotとdraft後のtarget状態復元を削減。履歴容量を初期設定の最大先読み数に合わせた。
  expert cache予算への自動再配分は未実装。
- 実モデルの変更前後を各条件5回比較。先読み1は37.29から37.60 tok/s、先読み3は23.81から23.92 tok/s。
  生成列は全15ケース一致。メモリは約109 MiB削減したが、大幅な速度改善やMTP無し超えには未到達。
- 既存FlashInfer backendはタイル選択エラーで初期化に失敗し、速度比較は未成立。Tritonを維持。
- 下記4のexpert転送overlapを実装してMTPで既定有効にした。
  [第2段階の実測](../benchmarks/results/mtp-20260910-overlap/implementation.md)では、各条件5回の中央値で
  先読み1が37.02から38.19 tok/s、先読み3が23.93から24.51 tok/s。生成列は全15ケース一致。
  大幅な改善やMTP無し超えには未到達。
- 下記3のgreedy draft + stochastic targetを切り替え可能な経路として実装。
  [第3段階の実測](../benchmarks/results/mtp-20260910-greedy-draft/implementation.md)では、各条件5回の中央値で
  先読み1が38.16から34.96 tok/s、先読み3が24.33から23.39 tok/sへ低下。
  採用率低下があり、このモデル・設定では推奨せず、既定無効を維持する。
  この段階では適応制御は未実装だった。
- draft-only QSA索引再利用も実装し、既定無効の実験オプションを追加した。
  [第4段階の実測](../benchmarks/results/mtp-20260910-qsa-reuse/implementation.md)では、短い入力の先読み3は
  24.39から24.37 tok/sで横ばい、生成列は全15ケース一致。3,661トークンの入力では各3回の中央値が
  23.86から25.28 tok/sへ約6%改善したが、確率的生成列も変わる比較であり、一般的な改善とは断定しない。
  MTP無し超えには未到達。
- 実時間による0/1/2/3の自動選択、disk PLEの非同期化、実モデル/API/OpenWebUIの検証を実施した。
  [最新の実装・検証結果](../benchmarks/results/mtp-auto/implementation.md)を参照。
  オフライン各5回の中央値はMTP無し46.82、自動45.03 tok/s。自動は先読み1を試して0へ戻るため、
  この速度をMTP本体の速度とは数えない。HTTP各3回は47.62と45.09 tok/sで、差は約5.3%。
  非同期PLEは固定1で約2%、固定3で約3%改善し、全10組の生成列が一致した。
  369テストと8K/32K境界、同時greedy生成、stop/切断復帰を検証済み。
  次の課題はtarget検証とexpert移動のコスト削減。MTPを継続したままMTP無しを上回る結果はまだない。
- [targetボトルネックの追加調査](../benchmarks/results/mtp-target/implementation.md)も実施した。
  PCIe 4.0 x16上のexpertコピーは約25.5 GB/s。起動構成の調整は採用せず、
  減衰付き頻度キャッシュと、一部BF16層の行ごとの並行実行を任意設定として実装。
  前者は固定1で約1.1%、後者は固定3で約1.6%の中央値改善に留まるため、既定無効を維持する。
  大きなGDN入力射影とexpert転送が残っており、MTP無し超えには未到達。
- [expert転送削減とGDN重み共有](../benchmarks/results/mtp-reuse/implementation.md)を追加した。
  expert側は減衰付き頻度に次の層までの距離を加えたeviction制御、GDN側は複数行で
  BF16重みを共有する固定FP32 reductionを実装。GDNは通常decodeにも同じ演算を適用する。
  従来cuBLASとは丸め差があるため、いずれも既定無効の選択式として検証する。
  両方有効時の各3回の中央値は、先読み1が34.78から37.05、先読み3が24.72から28.51 tok/s。
  expert単独では生成列を保ち、推定転送量を先読み1で7.3%、先読み3で6.0%削減した。
  GDNの変更は従来とのsampling生成列・分布に差があり、MTP無しでは速度が下がるため既定有効化しない。
  新GDN同士ではMTP有無の生成一致、8K境界、0/1/2/3切替を確認。MTP無し超えには未到達。

## 方針

まず固定先読み1トークンの実処理を安くする。既存のNVFP4 backendを比較し、不要な
状態コピー、expert転送の待ち、ドラフトのsampling負荷を順に削減する。
適応制御は低速化を避けるために併設するが、MTPを停止して得た速度をMTP本体の高速化とは数えない。

ターゲットのsampling設定は現在のtemperature=1、top_k=20、top_p=0.95を保つ。
ターゲットの計算順を保つ変更、ドラフト分布だけを変える変更、ターゲットの数値計算が
変わるbackend実験を、それぞれ検証する。

## 比較の前提

[前回の同条件実測](../benchmarks/results/mtp-20260909/chat-performance.md)は以下。
512出力トークン、2回平均、128トークンwarmup、61入力トークン、
1000128トークン容量、15629 KVページ、3073 expert slots。

| 先読み | 既定の計算順 | 任意のbatched BF16 |
|---|---:|---:|
| 0 | 47.68 tok/s | 48.03 tok/s |
| 1 | 36.29 tok/s | 39.96 tok/s |
| 3 | 24.17 tok/s | 30.07 tok/s |

1Mは予約容量であり、実際の入力長ではない。API検証の42-44 tok/sは自動割当3475 slotsで、
この表と同じcache条件ではない。今後も測定条件を混ぜない。

任意BF16経路の集計から概算すると、先読み1の1サイクルは約39 msで約1.56トークンを確定する。
48 tok/sを上回るには約32.4 ms以下、つまり約17%のサイクル時間削減が必要。
先読み3は約64 msで約1.92トークンのため、約40 ms以下、約37%の削減が必要になる。
これは末尾の例外処理を含む集計からの概算であり、直接測ったサイクル時間ではない。
これが先読み1を最初の対象にする理由である。

## 他ランタイムから分かったこと

### llama.cpp

Generic MTPは[PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)でマージ済み。
一方、今回のモデル系列のMTPを追加する[PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243)は
調査時点でopen / Draft / 未マージ。generic MTP対応と、このcheckpointでの動作確認は分ける。

`common_speculative_impl_draft_mtp::draft`はtop-1候補を選び、`p_min`によって先読みを打ち切る。
backend samplingへ渡す経路もある。ターゲット側の`common_sampler_sample_and_accept_n`は
通常のsampling結果とdraft IDが一致するprefixを採用する。
これは現在のFreeTokenの確率的proposalとp/q検証とは異なる方式だが、出力までgreedyにする
必要はない。[draft実装](https://github.com/ggml-org/llama.cpp/blob/4b98ab805a2638121f1671bf572832e07ef13e7d/common/speculative.cpp#L1324)、
[target sampling](https://github.com/ggml-org/llama.cpp/blob/4b98ab805a2638121f1671bf572832e07ef13e7d/common/sampling.cpp#L678)

recurrent stateは保存済み状態のindexを選んで巻き戻す。
FreeTokenの中間状態保存は既に同じ方向にあるので、次の候補はコピー量と必要な保存段数の削減。
[recurrent rollback](https://github.com/ggml-org/llama.cpp/blob/4b98ab805a2638121f1671bf572832e07ef13e7d/src/llama-memory-recurrent.cpp#L161)

### SGLang

MTPは直列ドラフトと一括検証を使う。v2実装はaccepted lengthをGPU上で状態commitや
次回draft準備へ渡す。QSAではdraft-extendで得た索引を後続draftに再利用する経路もある。
共有対象はdraft backendだけで、targetのQSA索引は通常どおり計算する。
QSA共有はchain幅1・複数stepなどの条件付きで、現在はadaptiveとの併用を無効化している。
[v2 worker](https://github.com/sgl-project/sglang/blob/ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4/python/sglang/srt/speculative/eagle_worker_v2.py#L390)、
[verify処理](https://github.com/sgl-project/sglang/blob/ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4/python/sglang/srt/speculative/eagle_worker_common.py)

adaptiveの現行ソースは、採用長のEMA、warmup、更新間隔、hysteresisを持ち、
0トークンへの降格と再probeにも対応する。ただし実際のexpert転送時間を使う判断ではない。
FreeTokenへは時間当たりの確定トークン数を目的にした制御として取り入れる。
[adaptive実装](https://github.com/sgl-project/sglang/blob/ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4/python/sglang/srt/speculative/adaptive_spec_params.py#L142)

ReplaySSMは大きな中間GDN状態の代わりにcompactな入力を保持し、採用prefixの状態を再構成する。
[PR #28695](https://github.com/sgl-project/sglang/pull/28695)はマージ済みだが、Qwen4-ExpのPLE状態commitに
関する[修正PR #37794](https://github.com/sgl-project/sglang/pull/37794)はopen。
そのまま移植せず、GDN・PLE・QSAをまとめて正しさを確認する必要がある。

### vLLM

`draft_sample_method`の既定値はgreedy。確率的draftも選べるが、後者はdraft確率の保持を伴う。
greedy draftではproposalをone-hot分布として扱い、target側の確率的samplingを維持する。
[設定](https://github.com/vllm-project/vllm/blob/9ffb8cea96369e2164af184a10faaa618eaf6b92/vllm/config/speculative.py#L588)、
[rejection sampler](https://github.com/vllm-project/vllm/blob/9ffb8cea96369e2164af184a10faaa618eaf6b92/vllm/v1/sample/rejection_sampler.py#L816)

Qwen4-Exp MTPは、最終mixer前の複数hidden streamを次のdraftに渡し、LM headにはcollapsed streamを渡す。
同じMTP layerの反復とQSA索引再利用も実装されている。FreeTokenのhiddenの受け渡しは概ね同じ構造であり、
「MTP=3なら独立した3ヘッドを並列実行できる」という前提にはできない。
[Qwen4-Exp MTP](https://github.com/vllm-project/vllm/blob/9ffb8cea96369e2164af184a10faaa618eaf6b92/vllm/models/qwen4_exp/nvidia/mtp.py#L262)

動的先読みはbatch size別の0を含む深さ表を持つが、公式資料の検証対象はEAGLE/EAGLE3/DFlash。
さらに時間と期待採用数で検証量を決めるAdaptive Verificationは、現時点でconfidence head付きDSpark限定。
これらをnative MTPでそのまま利用できるとは扱わない。
[動的先読み](https://github.com/vllm-project/vllm/blob/9ffb8cea96369e2164af184a10faaa618eaf6b92/docs/features/speculative_decoding/dynamic_speculative_decoding.md)、
[Adaptive Verification](https://github.com/vllm-project/vllm/blob/9ffb8cea96369e2164af184a10faaa618eaf6b92/docs/features/speculative_decoding/adaptive_verification.md)

対象checkpointの作者はvLLMで161.12 tok/sを報告しているが、RTX PRO 6000 96 GB、131k容量、
BF16 KV、PLEのみCPU offloadという条件である。32 GB上でexpertを転送する今回の環境の目標速度に
直接流用しない。これは作者の報告で、こちらで再現した結果ではない。
[モデルカード](https://huggingface.co/aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP#qualified-runtime)

## 現在のFreeTokenで既にできていること

- target/draft CUDA Graph、QSAのactive-context範囲への縮小、GPU acceptance処理。
- 不採用後のtarget全体の再実行を省き、保存したGDN/PLE/QSA状態から採用prefixをcommit。
- CPUからGPUへの同一expert転送の重複排除。新規提案の対象は転送後の重み再読み出しや待ち時間。
- draft修復ではtargetの確定したhiddenを用い、次回の第1候補を先に準備している。
- `LMHead.forward`はprefill形式のdraft-extendで最終行を選んでから語彙射影している。
  「全catch-up行のLM headを最後の1行へ減らす」は既に実施済み。

確認箇所: [MTP runner](../python/freetoken/scheduler/mtp.py)、
[LM head](../python/freetoken/layers/embedding.py)、[expert管理](../python/freetoken/moe/offload_kernels.py)。

## 実施順と採用条件

### 1. 計測を拡張し、既存backendを先に比較する

`bench_mtp_chat.py`とMTP runnerの診断モードに以下を追加する。

- サイクル別の先読み長・採用prefix長・確定数・decode時間。深さ別の到達率も記録する。
- draft lookahead、target verify、draft修復、状態保存/commit、sampling、PLE lookup、CPU待ちの内訳。
- targetとdraft別のexpert hit/miss、転送byte、行ごとのexpert数と全行のunion、行間のexpert重複率。
- profilerの重複する親子時間を加算しない。prefillと初回graph captureはdecodeの内訳から分離する。

並行して`--nvfp4-backend flashinfer`を既存native経路とA/Bする。
現在の自動選択は主にM=1の形状基準であり、MTP検証のM=2/4とは最適点が違う可能性がある。
この経路は既に実装済みなので、新規kernel作成より先に試せる。
[選択条件](../python/freetoken/moe/nvfp4_backends.py)、[dispatch](../python/freetoken/layers/moe.py)

同じ実行で0/1/3を比べ、cache slots固定の比較と総VRAM予算固定の比較を記録する。
backendのpacking・丸め方が変わるため、これは数値差を伴い得る実験として扱う。
同一重みのnative/prepared表現を無条件に二重常駐させる設計は採らない。

この段階で、MTP入出力の1トークンずれ、pre-mixer hidden、norm、QSA/PLE cache境界をvLLMの
同系列実装と照合する。自由生成の採用率31%だけを根拠に実装ミスとは断定しない。
必要なら固定prefix・同じhiddenを使うMTP head単体の数値比較を加える。

### 2. 不要な状態コピーと固定メモリを削減する

`_snapshot/_restore`は各サイクルで約108 MiBの全GDN recurrent stateを複製し、
複数draftの後にも復元している。今回のMTP headはQSAであり、この段階でtarget GDNを更新しない。
さらにhistory経路の`_commit_prefix`は、snapshotした古いrecurrent tensorを参照していない。

draft専用QSA状態とtargetのcommit用状態の保存を分ける。tracked verificationでは
old conv/PLE/ngram/ringのみを保存し、unknown stateやfallback経路では全snapshotを残す。
繰り返し確保するmetadataや小tensorも再利用する。

historyの段数は固定5から設定上必要な最大K+1へ。固定K=1なら理論上540から216 MiBへ減らせる。
adaptiveで最大K=3を許す場合は4段が必要で、216 MiBのままにはできない。
CUDA Graphの参照先を途中で交換せず、初期化・cache budget・再captureの責任を明確にする。
解放したメモリをexpert cacheへ回す効果は、コピー削減そのものの効果と分けて測る。

採用条件: 出力/状態の一致、全採用長・nonzero slot・graph再生・fallback・cache rebuildの通過。
これ単独で48 tok/sを超えるとは見込まず、確実な無駄の削減として先行する。

### 3. ドラフトを軽くする

最初に「greedy draft + 確率的target」の切替を試す。ユーザーのtemperature/top_k/top_pは維持し、
draftのsoftmax/filter/乱数生成/full-vocabulary q保持を省く。

proposalを`q(y)=1`の決定的な候補とし、採用確率を`p(y)`、不採用時を候補y以外のpを
正規化した分布にする。あるいはtargetから通常samplingし、draftとの一致prefixを採用する。
greedy proposalなのに元のsoftmax qをp/q判定へ渡す実装にはしない。
これはtarget計算を変えずに設計できるが、現在の確率的proposalより採用率が下がる可能性がある。
判定基準は採用率ではなく、確定トークン当たりの時間とする。

次に複数先読み用のdraft-only QSA索引共有を独立した実験として追加する。
draft-extendの索引と後続causal tailを保持し、target側の選択は変えない。
qは実際に使ったdraftの分布と一致させる。圧縮境界、ring wrap、K変更、graph shape変更で
索引を更新/無効化する。K=1では省ける後続draftがないため、主な対象はK>=2と長い会話になる。

### 4. expert転送と計算の待ちを削減する

`Qwen4ExpMoE.forward`はrouter logitsを計算した後、shared expertを終えてから
routed expertの確保・転送・計算を行う。この順序を分割し、routerが確定した時点で不足expertの
転送を開始して、同じ層のshared expert/gate計算と重ねる。
次の層の未知のroutingを予測する必要はない。
[具体的な分割位置](../python/freetoken/models/qwen4_exp/moe.py)

転送完了event、使用中slotの保護、入力tensorの寿命、capture/replay、abort/rebuildを設計する。
現行copy kernelはGPU資源も消費するため、別streamにしただけで速くなるとは限らない。
常駐率が高い場合・低い場合の両方で、全体のms/tokenが下がることを採用条件にする。

計測でcache汚染が確認できた場合のみ、draft head用の小さなquotaやtarget優先のadmissionを試す。
総slot数を固定し、512個のdraft expertを全固定してtarget cacheを圧迫する案は先行しない。
行間のexpert重複が十分ある場合には、同じresident重みを複数行で使うnative NVFP4 kernelを検討する。
これは転送の重複排除とは別の最適化である。

### 5. 実時間で0/1/2/3を選ぶ適応制御

計測基盤ができた時点から設計できる。最初はK=1を候補とし、過去の確定済みサイクルの
`総時間 / 確定トークン数`、採用prefixの分布、cache miss、context bucketを使って選ぶ。
EMA・最低試行数・hysteresisを設け、graph capture時間を定常コストへ混ぜない。

目的関数は概念的には
`(draft + verify + commit + draft修復 + host待ち) / (1 + 期待採用数)`。
深い候補は、追加の確定数が追加コストに見合う場合だけ使う。

K=0ではlookaheadだけでなく次回第1draftの生成も止める。初版は一度停止したらそのリクエスト中は
停止を維持し、次のリクエストで再評価する。途中再開はdraft KVが古くなるため、別段階で
target hidden/IDのcatch-upまたは明示的resetを実装してから行う。

確率的proposalの打切りは、過去の履歴やsample前の分布統計で決める。
sampleしたyのq(y)が低いときだけ破棄するとproposal分布が条件付けされるため、元のqを
そのまま検証に使う方式は禁止する。

採用条件: 遅いpromptでMTPの無駄が縮小し、速いpromptで効果が残ること。
全体速度、MTP稼働中速度、K=0比率、探索コストを別々に報告する。

### 6. 計測結果を見て進める後続候補

| 候補 | 着手条件と範囲 |
|---|---|
| GPU上のcommitと次回準備 | `.item()`後のCPU待ちが支配的な場合。固定幅token/有効長/残差選択をGPUで処理し、page解放や出力を後から回収する。PLE disk lookupはCPUにIDを必要とするため、CPU依存ゼロを前提にしない。 |
| state index切替 / ReplaySSM | 状態コピーまたはVRAMによるexpert cache制約が残る場合。GDNだけの改善にせずPLE/ngram/QSAを含めて検証する。追加のfold計算との交換条件を測る。 |
| draft語彙shortlist | LM headの時間が大きい場合。targetは248320語を維持し、draftのみ縮小。日本語・英語・code・記号のcoverage、qの正規化/ID mapping/残差分布を検証する。 |
| KV予約とexpert cacheの配分 | 1M容量を維持した比較とは別枠で64k/128k予約を試す。予約削減で得た効果をMTP kernelの改善と混同しない。既存設定は勝手に変更しない。 |
| request間の一括検証・prefix cache | 同時チャットや長い複数turnのTTFTを改善する段階。61トークンの新規単独チャットの主対策にはしない。 |
| 並列draft・学習し直したhead | native1層MTPを設定変更だけで並列化しない。対応する重みの学習/取得と品質評価が必要な別プロジェクト。 |

draft語彙削減の参考には同系列モデルの[llama.cpp実験報告](https://github.com/ggml-org/llama.cpp/discussions/28512)があるが、
別hardware・実験forkの著者測定であり、FreeTokenの改善率予測には使わない。
vLLMも[並列draftには対応する学習済みモデルを使用](https://github.com/vllm-project/vllm/blob/9ffb8cea96369e2164af184a10faaa618eaf6b92/docs/features/speculative_decoding/parallel_draft_model.md)している。

## 検証と完了基準

最初の合格目標は固定K=1のMTP稼働中速度が、同時点のMTP無しを安定して上回ること。
10%以上の改善を実用上の目標とするが、達成予測ではない。
autoは別に、低採用率のケースで通常decodeに対する低速化を5%以内へ抑えることを目標にする。

- 主測定: 指定の猫prompt、sampling既定値、固定1M容量、512出力、warmup後5回以上。
  mode順を入れ替え、中央値・ばらつき・TTFT・VRAM・採用長・転送byteを残す。
- 回帰測定: 日本語自由文、英語、code、8k/32k以上の会話、greedyとsampling、1/2並行request。
  cold/warmとfirst-token以降の配信速度を分け、最後にOpenAI互換SSE APIで確認する。
- 正しさ: 全採用/初回不採用/中間不採用、EOS/stop/max_tokens、分割prefill、nonzero slot、
  QSAページ・圧縮・ring境界、abort、cache再構築、K切替を確認する。
- sampling: 小語彙の分布収束試験、ゼロ確率とtop-k/top-p境界、greedy draftのone-hot proposal、
  shortlist外tokenへの補正を確認する。同じseedの文章一致を分布保存の証明にはしない。
- 数値計算を保つ変更はgreedyと状態の一致を確認する。backendを変える実験はlogit/分布差と
  実モデル出力を評価し、既定経路へ黙って混ぜない。

着手する最初のまとまりは、**計測追加 + 不要snapshot削減 + 既存NVFP4 backend比較**。
その結果で、転送overlapとドラフト軽量化のどちらを主な開発対象にするか決める。
