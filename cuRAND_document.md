# cuRANDのドキュメント
https://docs.nvidia.com/cuda/curand/index.html

(↑を元に訳した、英弱かつ情弱奴が自分用に作ったものなので
正確な内容は原文を見ましょうfkt)

# 始めに

cuRANDライブラリは擬似乱数(pseudorandom)<br>
・準乱数(quasirandom)の高速な生成を行う手段を提供する。<br>
疑似乱数とは、真の乱数の要件をほとんど満たすが、<br>
決定論的なアルゴリズムによって作られる数列のこと。<br>
*n*-次元中の準乱数的点列とは、決定論的なアルゴリズムによって作り出される数列で、<br>
*n*-次元空間に均一に分布するような点列のこと。<br>
cuRANDは２つの部分からなる。(ホスト(CPU)側のライブラリ、デバイス(GPU)側のヘッダファイル)<br>

１つ目はホスト側のライブラリである。<br>
cuRAND以外のライブラリと同様に、`/include/curand.h`を<br>
インクルードすることで関数の宣言とライブラリへのリンクを行うことができる。<br>
乱数生成はホスト上でもデバイス上でも行える。<br>
デバイスでの乱数生成は、ホストでライブラリを呼び出すことで実行される。<br>
デバイスで生成された乱数はデバイスのグローバルメモリに記録される。<br>
自作カーネルを呼び出してこの乱数を使っても良いし、ホストにコピーすることもできる。<br>
ホストでの乱数生成は、すべての処理がホスト上で行われ、<br>
生成された乱数もホストのメモリに記録される。<br>

２つ目が部分デバイス側のヘッダファイル`/include/curand_kernel.h`である。<br>
このファイルの中で定義された関数は、<br>
デバイス上の乱数生成器の状態を決めて乱数列を作り出すためのものである。<br>
このヘッダファイルをインクルードし、<br>
自作カーネルでこのヘッダファイルで定義された関数を呼び出すこともできる。<br>
このようにすると、グローバルメモリに書き込み読み込むという作業なしに<br>
生成された乱数を使うことができる。<br>

# 1.互換性とバージョン
ホスト側のcuRANDのAPIは後方互換性を持つように作られている。<br>
いまcuRANDを使っているプログラムは、cuRANDをアップデートしてもそのまま動き続けるはずだ。<br>

一方バイナリのレベル(どう訳すのが適切？fkt)では互換性は保証されない。<br>
(この先の訳わからんfkt)<br>
(ホスト側は互換性がないかもしれないがデバイス側はほとんどの場合問題なかろう？fkt)<br>

# 2.ホストのAPI概略
`curand.h`をインクルードせよ。<br>
CUDAランタイムを使え。<br>

乱数を作り出すのは生成器である。<br>
cuRANDの中の生成器は乱数列を作るのに必要なすべての内部状態を含んでいる。<br>
乱数生成の手順は通常次のようなものである。<br>

1. `curandCreateGenerator()`でほしい種類の(後述)乱数生成器をつくる。
1. 生成器のオプションを指定する(シードなど)。
1. `cudaMalloc()`でデバイス上にメモリを確保する。
1. `curandGenerate()`などの関数で乱数を作る。
1. 生成された乱数を使う。
1. `curandGenerate()`を追加で呼び出してもっと乱数を作ることもできる。
1. `cuandDestroyGenerator()`で後始末。

ホストで乱数を作るときは、上の手順の1.で`curandCreateGeneratorHost()`を<br>
`curandCreateGenerator()`の代わりに呼び出す。<br>
また、手順3で乱数を収めるメモリをホスト上に確保する。<br>
それ以外は同じである。<br>

同時に複数の乱数生成器を作ることも可能である。<br>
複数の生成器はそれぞれ異なる内部状態を持ち、互いに独立である。<br>
それぞれの生成器から作られる乱数列はいずれも決定論的である。<br>
乱数生成のためのパラメタが同じであれば、何度プログラムを走らせても同じ乱数列が生成される。<br>
デバイスで作る乱数列とホストで作る乱数列は同じものである。<br>

注意。上記手順4.における`curandGenerate()`は非同期的にカーネルを立ち上げ値を返す。<br>
curandGenerate()の結果を必要とするカーネルが同時に動いているときには、<br>
`curdaThreadSynchronize()`を呼ぶとかストリームを管理する手続きを踏むとかして、<br>
件のカーネルが立ち上がる前に乱数生成器が実行終了するようにしなければならない。<br>

注意。ホストのメモリのポインタをデバイスで動いている生成器に渡すとか、<br>
デバイスのメモリのポインタをホストで動いている生成器に渡すとかはできない。<br>
このとき、関数の動作は未定義である。<br>

## 2.1.生成器の種類
乱数生成器は9種類ある。<br>
`curandCreateGenerator()`に以下の型名を渡すことで、乱数生成器が作られる。<br>

|名前|種類|備考|
|:--|:--|:--|
|`CURAND_RNG_PSEUDO_XORWOW`|疑似乱数|XORWOW(xor-shift familyの一種)を使う。
|`CURAND_RNG_PSEUDO_MRG32K3A`|同上|Combined Multiple Recursive(とは？fkt)の仲間。
|`CURAND_RNG_PSEUDO_MTGP32`|同上|Mersenne Twisterの仲間、GPU用にパラメタが調整されている。
|`CURAND_RNG_PSEUDO_PHILOX4_32_10`|同上|Philox familyの仲間。(詳しい説明わからんfkt)
|`CURAND_RNG_PSEUDO_MT19937`|同上|Mersenne Twister familyの仲間。CPU用のと同じパラメタだが数列の順序が異なる。<br>ホストのAPIだけをサポートする (とは？fkt)。sm\_35以降のアーキテクチャで使える。
|`CURAND_RNG_QUASI_SOBOL32`|準乱数|Sobol数列(とは？fkt)を作る。32bit版。
|`CURAND_RNG_QUASI_SOBOL64`|同上|Sobol数列を作る。64bit版。
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`|同上|scrambled Sobol数列(とは？fkt)を作る。64bit版。
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`|同上|scrambled Sobol数列を作る。32bit版。

## 2.2.生成器のオプション
作られた生成器を定義するのには、いずれの種類にも共通な3つのオプションを指定する。<br>

### 2.2.1.シード(seed)
乱数生成器を初期化する64bitの整数。<br>
生成器に同じシードを与えれば、同じ数列が作られる。<br>

### 2.2.2.オフセット(offset)
各生成器の用いるアルゴリズムによって作り出される数列の、<br>
最初のいくつかを飛ばすという指定。<br>
offset = 100とすれば、アルゴリズムによって作り出された数列の1から99番目は飛ばし、<br>
100番目とその先の数が乱数生成器から出てくる。<br>

注意。`CURAND_RNG_PSEUDO_MTGP32`と`CURAND_RNG_PSEUDO_MT19937`は<br>
オフセットオプションを使えない。<br>

### 2.2.3.格納順(order)
GPUの各スレッドで作られた乱数を<br>
どのような順序でグローバルメモリに保存するかという指定。<br>
以下の5種類がある。<br>

|名前|種類|備考|
|:--|:--|:--|
|`CURAND_ORDERING_PSEUDO_DEFAULT`|疑似乱数用|
|`CURAND_ORDERING_PSEUDO_BEST`|同上|MT19937以外では`CURAND_ORDERING_PSEUDO_DEFAULT`と同じ結果になる。<br>cuRANDのアップデートによって`CURAND_ORDERING_PSEUDO_BEST`の<br>速度と精度を高める予定(20200113)。<br>ただし、乱数が決定論的なものであり<br>プログラムを走らせるたびに同じ結果になるという点は変えない。
|`CURAND_ORDERING_PSEUDO_SEEDED`|同上|
|`CURAND_ORDERING_QUASI_DEFAULT`|準乱数用|
|`CURAND_ORDERING_PSEUDO_DEFAULT`|疑似乱数用|すべてのcuRANDのバージョンで同じ結果になることが保証されている。

(乱数生成器の種類と格納順の組み合わせごとの振る舞いが述べられているが、略fkt)<br>

- MTGP32
	- `CURAND_ORDERING_PSEUDO_BEST`

		今の所(20200113)`CURAND_ORDERING_PSEUDO_DEFAULT`と同じ振る舞い。<br>
	- `CURAND_ORDERING_PSEUDO_DEFAULT` 

		MTGP32の裏では192個の乱数生成器が動いており、<br>
		それぞれがMTのアルゴリズムで使う相異なるパラメタを持っている。<br>
		MTGP32が作り出す乱数列は、<br>
		192個の生成器それぞれが作り出す乱数を256個ずつ並べて作られる。<br>
- MT19937
	- `CURAND_ORDERING_PSEUDO_DEFAULT` 

		(よくわからんfkt)<br>
	- `CURAND_ORDERING_PSEUDO_BEST`

		速度を上げるために`CURAND_ORDERING_PSEUDO_DEFAULT`からの変更点がある<br>
		乱数列の格納順はアーキテクチャ(sm_\*\*の\*\*の所？fkt)ごとに異なる。<br>
		乱数生成の方法は`CURAND_ORDERING_PSEUDO_DEFAULT`と同じだが<br>
		裏で動く生成器の個数が違う。<br>
		"generate seed"(シードの生成？fkt)は`CURAND_ORDERING_PSEUDO_DEFAULT`より<br>
		こちらのほうがかなり速い。<br>




















