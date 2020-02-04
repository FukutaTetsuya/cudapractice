# cuRANDのドキュメント
https://docs.nvidia.com/cuda/curand/index.html

(↑を元に訳した、英弱かつ情弱奴が自分用に作ったものなので
正確な内容は原文を見ましょうfkt)

デバイスAPIのMTGP32のあたりで、
状態ベクトル(state array)とパラメタの組(parameter set, sequence)を混同している模様。

cudaHostAlloc()は濫用すると遅くなるらしいので、
頻繁にデバイスから参照されるような配列だけに使うのがよかろう。

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

乱数を作り出すのは乱数生成器である。<br>
cuRANDの中の生成器(curandGenerator\_t型で宣言される、乱数生成器として扱われる何かfkt)は<br>
乱数列を作るのに必要なすべての内部状態を含んでいる。<br>
乱数生成の手順は通常次のようなものである。(2.5.に実例)<br>

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
|`CURAND_ORDERING_PSEUDO_DEFAULT`|疑似乱数用|すべてのcuRANDのバージョンで同じ結果になることが保証されている。
|`CURAND_ORDERING_PSEUDO_BEST`|同上|MT19937以外では`CURAND_ORDERING_PSEUDO_DEFAULT`と同じ結果になる。<br>cuRANDのアップデートによって`CURAND_ORDERING_PSEUDO_BEST`の<br>速度と精度を高める予定(20200113)。<br>ただし、乱数が決定論的なものであり<br>プログラムを走らせるたびに同じ結果になるという点は変えない。
|`CURAND_ORDERING_PSEUDO_SEEDED`|同上|
|`CURAND_ORDERING_QUASI_DEFAULT`|準乱数用|

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

		格納の順序はCPU上でのMT19937の実装に強く依存している。(とはfkt)<br>
		MT19937を使ったカーネルの裏では8192個の独立な乱数生成器が動いている。<br>
		オリジナルのMTが作り出す乱数列の2^{1000}要素ずつを<br>
		それぞれの生成器が作り出す。<br>
		またそれぞれの生成器は8個ずつ乱数を作り出す。<br>
		速度を上げるため格納順はオリジナルのMTとは異なる。<br>
		格納順は、ハードウェアに依存しない。<br>
	- `CURAND_ORDERING_PSEUDO_BEST`

		速度を上げるために`CURAND_ORDERING_PSEUDO_DEFAULT`からの変更点がある<br>
		乱数列の格納順はアーキテクチャ(sm_\*\*の\*\*の所？fkt)ごとに異なる。<br>
		乱数生成の方法は`CURAND_ORDERING_PSEUDO_DEFAULT`と同じだが<br>
		裏で動く生成器の個数が違う。<br>
		"generate seed"(シードの生成？fkt)は`CURAND_ORDERING_PSEUDO_DEFAULT`より<br>
		こちらのほうがかなり速い。<br>

## 2.3.戻り値
ホストのcuRANDライブラリはすべて、`curandStatus_t`型の戻り値を返す。<br>
エラーなく呼び出せた場合、`CURAND_STATUS_SUCCESS`を返す。<br>
エラーが起こった場合、エラーに応じて`CURAND_STATUS_SUCCESS`とは異なる値を返す。<br>
CUDAではカーネル(デバイスを動かす関数かfkt)が<br>
CPUのコードとは非同期的に実行される。<br>
そのため、cuRANDの関数を走らせている間に<br>
cuRANDではないカーネルでエラーが生じることがある。<br>
この時、cuRANDの関数は`CURAND_STATUS_PREEXISTING_ERROR`を返す。<br>

## 2.4.乱数を生成する関数(generation functions)
(色々作れるが、全部はめんどいので書かない。詳細はこのファイルの最上部にあるリンクからfkt)<br>
ランダムなビット列、一様分布、正規分布、対数正規分布、ポアソン分布を作れるらしい。<br>

--------------------
```
curandStatus_t 
curandGenerate(
    curandGenerator_t generator, 
    unsigned int *outputPtr, size_t num)    
```
`curandGenerate()`は戻り値が`curandStatus_t`型である。<br>
(第一引数は`curandGenerator_t`型変数で、乱数を作らせたい乱数生成器の名前を入れる。fkt)<br>
`curandGenerate()`で使える乱数生成器の種類は(2.1.参照fkt)、<br>
XOROWOW, MRG32k3a, MTGP32, MT19937, Philox\_4x32\_10, SOBOL32。<br>
(第二引数はデバイス上の`uns int`型配列へのポインタである。fkt)<br>
出力されるのは、全てのビットがランダムな32bit uns intの数列である。<br>
(第三引数は`size_t`型変数で、生成する数列の要素数を指定する。fkt)<br>

--------------------
```
curandStatus_t 
curandGenerateLongLong(
    curandGenerator_t generator, 
    unsigned long long *outputPtr, size_t num)
```
`curandGenerateLongLong()`は戻り値が`curandStatus_t`型である。<br>
(第一引数は`curandGenerator_t`型変数で、乱数を作らせたい乱数生成器の名前を入れる。fkt)<br>
`curandGenerateLongLong()`で使用できる生成器の種類は、SOBOL64である。<br>
(第二引数はデバイス上の`uns long long int`型配列へのポインタである。fkt)<br>
出力されるのは、全てのビットがランダムな64bit uns long long intの数列である。<br>
(第三引数は`size_t`型変数で、生成する数列の要素数を指定する。fkt)<br>

--------------------
```
curandStatus_t
curandGenerateUniformDouble(
    curandGenerator_t generator,
    double *outputPtr, size_t n)
```
`curandGenerateUniformDouble()`は戻り値が`curandStatus_t`型である。<br>
(第一引数は`curandGenerator_t`型変数で、乱数を作らせたい乱数生成器の名前を入れる。fkt)<br>
(使える生成器の種類は書いてないfkt)<br>
(第二引数はデバイス上の`double`型配列へのポインタである。fkt)<br>
出力されるのは、倍精度の一様乱数である。<br>
(`curandGenerateUniform()`同様、範囲は(0,1]と思われる。fkt)<br>
(第三引数は`size_t`型変数で、生成する数列の要素数を指定する。fkt)<br>

--------------------

同じ乱数生成関数をなんども呼び出して、いくつもの乱数列を得ることができる。<br>
擬似乱数の生成器では、なんども呼び出して得られた乱数列を繋げたものと<br>
いっぺんに作った同じ長さの乱数列の中身は同一である。<br>
準乱数については、(理由はよくわからんがfkt)異なる結果が得られる。<br>

倍精度を扱えるのは1.3世代以降のハードだけである。<br>

## 2.5.ホストAPIのサンプル
(自分なりに書いてみた、原文の方がいい感じに仕上げてあるfkt)<br>
```
/*
   *準乱数生成器のseedはどうやって与える？
   */
#include<stdio.h>
#include<time.h>
#include<curand.h>
#include<cuda.h>

int main(void) {
	float *d_array_rei;
	float *d_array_mari;
	float *h_array_rei;
	float *h_array_mari;
	size_t n = 16384;
	unsigned int dimension = 2;
	float mean = 0.0;
	float standard_deviation = 1.0;
	int i;
	//これが乱数生成器を指す名前的なもの
	//二つ作ってみる
	curandGenerator_t rei, mari;
	
	//デバイスとホストにメモリを確保する
	cudaMalloc((void **)&d_array_rei, n * sizeof(float));
	cudaMalloc((void **)&d_array_mari, n * sizeof(float));
	cudaHostAlloc((void **)&h_array_rei, n * sizeof(float), cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_array_mari, n * sizeof(float), cudaHostAllocMapped);

	//乱数生成器を作る
	//\->reiはXORWOWというアルゴリズムを使い擬似乱数を作る乱数生成器とする
	curandCreateGenerator(&rei, CURAND_RNG_PSEUDO_XORWOW);
	//\->mariはSOBOLというアルゴリズムを使い準乱数を作る乱数生成器とする
	curandCreateGenerator(&mari, CURAND_RNG_QUASI_SOBOL32);

	//乱数生成器にシードを与える。ULLは型、64bit符号なし整数
	//\->準乱数生成器を初期化する関数がわからないので
	//    mariはそのままつかう
	curandSetPseudoRandomGeneratorSeed(rei, 890106ULL);
	//\->time()を使うならこちら
	//curandSetPseudoRandomGeneratorSeed(rei, (unsigned long)time(NULL));

	//オフセットを伝える。これも64bit符号なし整数で指定する
	//\->reiにだけオフセットを設け、mariはオフセットなしとする
	curandSetGeneratorOffset(rei, 5ULL);

	//rei,mariに格納順を伝える、どちらもデフォルトでよかろう
	curandSetGeneratorOrdering(rei, CURAND_ORDERING_PSEUDO_DEFAULT);
	curandSetGeneratorOrdering(mari, CURAND_ORDERING_QUASI_DEFAULT);

	//準乱数については、何次元空間で均一に分布するかを指定できる
	curandSetQuasiRandomGeneratorDimensions(mari, dimension);

	//n個だけ乱数を作らせ、結果をd_arrayに収める
	//\->reiにはfloatの一様乱数をつくらせる
	curandGenerateUniform(rei, d_array_rei, n);
	//\->mariにはfloatの正規分布乱数をつくらせる
	curandGenerateNormal(mari, d_array_mari, n, mean, standard_deviation);

	//デバイスからホストへ生成された乱数を持ってくる
	cudaMemcpy(h_array_rei, d_array_rei, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_array_mari, d_array_mari, n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("rei's result\n");
	for(i = 0; i < 10; i += 1) {
		printf("%d:%f\n", i, h_array_rei[i]);
	}
	printf("mari's result\n");
	for(i = 0; i < 10; i += 1) {
		printf("%d:%f\n", i, h_array_mari[i]);
	}

	//乱数生成器を消す
	curandDestroyGenerator(rei);
	curandDestroyGenerator(mari);
	//デバイスとホストのメモリを解放する
	cudaFree(d_array_rei);
	cudaFree(d_array_mari);
	cudaFreeHost(h_array_rei);
	cudaFreeHost(h_array_mari);
	return 0;
}
```

## 2.6.静的ライブラリの使用について
わかんね<br>
コンパイルはこんな感じにすべき
```
nvcc -arch=sm_60 -lcurand genRandomNumber.cu
```

## 2.7.実行速度についての但し書き
乱数生成は、なるべく大きい塊で作るのがよい。ちょっとずつ作るのは効率が悪い。<br>

デフォルトの乱数生成器であるXORWOWをデフォルトの乱数格納順で呼び出すと、<br>
初回だけ立ち上げに時間がかかる。<br>
立ち上げに時間がかかるのを避けるには、格納順に`CURAND_ORDERING_PSEUDO_SEEDED`を指定する。<br>

乱数生成器にMTGP32を使う時、16384個ずつ乱数を作るのが一番速い。<br>
(これは仕様らしいが理由はよくわからんfkt)<br>

乱数生成器にMT19937を使う時、一番速いのは2GBより多いデータを作る時。<br>
最高速の8割を得られるのは、80MB以上のデータを作る時。<br>
(これは仕様らしいが理由はよくわからんfkt)<br>

乱数生成器にPhilox_4x32_10を使う時、一番速いのは(スレッド数x4)個の乱数列を作る時。<br>
1スレッドが4個の乱数をつくるから。<br>

# 3.デバイスAPIの概略
デバイスAPIを使うには、`curand_kernel.h`をインクルードする。<br>
この中にcuRANDのデバイスの関数が定義されている。<br>
擬似乱数の生成と準乱数の生成を行う関数が含まれる。<br>

## 3.1.擬似乱数列
ランダムビット列生成と、分布を指定した乱数列生成が可能である。<br>
### 3.1.1.XORWOW, MRG32k3aを使った生成器によるビット列生成
(略fkt)
### 3.1.2.MTGP32を使った生成器によるビット列生成
(アルゴリズムについて重大な勘違いがある気がするがよくわからないfkt)<br>
広島大で作られたコードを応用したもの(SAITO Mutsuo arxiv 2010)。<br>

>このアルゴリズムにおいて、サンプル(**R**からの？fkt)が<br>
>複数のシーケンスのために作られる。<br>
>それぞれのシーケンスは各自パラメタの組に基づいて行われる。<br>
>cuRANDは200個のパラメタの組を使う。<br>
>このパラメタの組は2^{11214}の周期を持つ<br>
>32bit乱数生成器のためにすでに用意されている。<br>
>異なるパラメタの組を使うこともできる。<br>
>一つのパラメタの組、すなわち一つのシーケンスごとに一つの<br>
>state structure(生成器の状態を収める構造体？fkt)が存在する。<br>
>MTGP32のアルゴリズムによってスレッドセーフに生成器の状態を更新し<br>
>乱数を生成することができる。<br>

(以下、読解が怪しいままの意訳)<br>
デバイスから呼ぶことができるように、
デバイス上に乱数生成器を作ろうということ。<br>
同じブロックに属する複数(最大256)のスレッドから<br>
同じ一つの乱数生成器を呼び出して使うことができる。<br>

乱数生成器の内部状態は1024個の整数の組で完全に決まる。<br>
乱数生成器の内部状態の更新のために必要なパラメタの組をシーケンスと呼ぶ。<br>
このシーケンスは200個の値の組である。<br>
シーケンスをひとまとめにして構造体に収めている？<br>
で、cuRANDはシーケンスのプリセットを用意してくれている。<br>
初期シーケンスを他の方法で用意することもできる。(MTGPに関する広島大の論文を見よ。)<br>
(以上、読解が怪しいままの意訳fkt)<br>

注意。異なるブロックから同じ状態パラメタを安全に操作することはできない。<br>
注意。同じブロックの中からでも、一組の状態パラメタを<br>
操作できるのは最大で256スレッドである。<br>

MTGP32を使ったデバイス上の乱数生成器に関してホストの関数が2つある。<br>
デバイスのメモリ上に置かれる複数のシーケンス<br>
(…の中身のパラメタの値)を設定するためのと、<br>
乱数生成器の初期内部状態を設定するための関数である。<br>

---
```
__host__ curandStatus_t curandMakeMTGP32Constants(mtgp32_params_fast_t params[],
						mtgp32_kernel_params_t *p)
//(原文では引数の型のアンダーバーが消えているが、ミスと思われfkt)
```
`curandMakeMTGP32Constants`は、<br>
`mtgp32_params_fast_t`型で用意しておいた初期シーケンス`params[]`を<br>
デバイス上の関数で使える`mtgp32_kernel_params_t`型に変換し、<br>
デバイス上のメモリ`p`にコピーする。<br>

---
```
__host__ curandStatus_t
curandMakeMTGP32KernelState(curandStateMtgp32_t *s,
				mtgp32_params_fast_t params[],
				mtgp32_kernel_params_t *k,
				int n,
				unsigned long long seed)
//(第三引数kがcurandMakeMTGP32Constantsと同じpではないのは何か含意がある？fkt)
```
`curandMakeMTGP32KernelState`は、`n`個の内部状態を<br>
ある一つのシーケンス`params[]`とシード`seed`に基づいて初期化し、<br>
その結果を`s`の指すデバイス上のメモリにコピーする。<br>
(乱数生成器が実際生成に使うシーケンスはsである？fkt)<br>
注意。プリセットのシーケンスを使う場合、`n`の最大値は200である。<br>

---

MTGP32を使ったデバイス上の乱数生成器でランダムなビット列を作るために、<br>
デバイスの関数が二つある。<br>

---

```
__device__ unsigned int
curand(curandStateMtgp32_t *state)
```

この関数はスレッドの番号(index)を計算し、<br>
そのスレッド番号に対して乱数を計算し、<br>
乱数生成器の状態(シーケンス)を更新する。<br>
スレッド番号`t`の決め方は、<br>
`t= (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x`<br>
である。<br>

この関数`curand()`は、以下の条件のもと、一度のカーネルの立ち上げで<br>
繰り返し呼び出すことができる。<br>
(`curand()`を使うようなデバイス関数の呼び出しを一度行うこと？fkt)<br>

- 256本以下のスレッドを持つブロックから呼び出すこと。<br>
- 一つのシーケンスはただ一つのブロックの中でのみ使うことができる。<br>
- 一つのブロックの中で複数のシーケンスを使うことができる。<br>
- コードの中のある点で、一つのブロックの中のスレッドは、<br>
	全てがこの関数を呼び出すか、全てがこの関数を呼び出さないかの<br>
	どちらかでなくてはいけない。<br>
	(if(t=odd){curand();} else {}みたいなことが許されない？fkt)<br>

---
```
__device__ unsigned int
curand_mtgp32_specific(curandStateMtgp32_t *state, unsigned char index,
			unsigned char n)
//(原文では関数名中のアンダーバーが消えているが、ミスと思われfkt)
```
この関数`curand_mtgp32_specific()`は、<br>
スレッド固有の場所を示す`index`に基づいて乱数を計算し、<br>
乱数生成器の内部状態のオフセットを`n`だけ進める。<br>
この関数`curand_mtgp32_specific()`を、以下の条件のもと<br>
一度のカーネルの立ち上げのなかで繰り返し呼び出すことができる。<br>

- 一つのシーケンスを同時に使うことができるスレッド数は256本までである。<br>
- 一つのブロックの中で、一つのシーケンスを`n`本のスレッドが呼び出すとする。<br>
	`index`は、`0..n-1`を走るものである必要がある。<br>
	このインデックスはスレッド番号と一致しなくても良い。<br>
	このインデックスは、`curand_mtgp32_specific()`を呼び出す関数の中で<br>
	スレッドごとに割り振る。<br>
	インデックス`0..n-1`は、それらの全てが使われるか、<br>
	それらの全てが使われないかのどちらかでなくてはいけない。<br>
- 一つのシーケンスは一つのブロックの中でしか使うことができない。<br>
- 一つのブロックの中で複数のシーケンスを使って乱数を作ることができる。<br>

---

(Figure 1の下で最大スレッド数が256である理由を説明しているが、略。
一度に256スレッドから呼び出すのが一番効率がいいらしい。fkt)


### 3.1.3.Philox_4x32_10を使った生成器によるビット列生成
(略fkt)
### 3.1.4.特定の分布をもつ乱数列(distribution)

---
```
__device__ float
curand_uniform(curandState_t *state)
```
`curand_uniform()`は(0,1]の一様乱数を返す。<br>
unsigned int型の乱数を作る元々の乱数生成器の結果をつかって<br>
(0,1]の一様乱数を作っている。<br>
一つの実数乱数を作るために<br>
元々の乱数生成器で作られる乱数をいくつ消費するかは、<br>
定まっていない。<br>

---

(他にもバリエーションがあるが略fkt)<br>

## 3.2.準乱数列
## 3.3.頭切り(skip-ahead)
## 3.4.デバイス上での離散分布の生成
## 3.5.性能についての注意書き
## 3.6.デバイスAPIのコードの例
(自分なりに書いてみた、原文の方がいい感じに仕上げてあるfkt)<br>

```
/*
   *デバイス上でMTGP32を使って乱数を生成してみる
   */
#include<stdio.h>
#include<time.h>
#include<cuda.h>

//デバイスAPIをインクルード
#include<curand_kernel.h>

//mtgp32を使う乱数生成器のセットアップに使うホストの関数をインクルード
#include<curand_mtgp32_host.h>

//mtgp32で使うシーケンス(パラメタの組)のプリセットをインクルード
//このヘッダの中に、mtgp32_params_fast_t型でプリセットのシーケンスが入っている
//プリセットの名前は`mtgp32dc_params_fast_11213`
#include<curand_mtgp32dc_p_11213.h>

//デバイスのカーネルを呼び出すときに指定するスレッド数・ブロック数
//スレッド数はmtgp32のデバイスAPIを使う限り、256以下でなくてはならない
#define NUM_THREAD 8
#define NUM_BLOCK 4

void check_cuda_funcs_executed_right(cudaError_t status) {
	if(status != cudaSuccess) {
		printf("error cuda funcs\n");
		exit(1);
	}
}

void check_curand_funcs_executed_right(curandStatus_t status) {
	if(status != CURAND_STATUS_SUCCESS) {
		printf("error curand funcs\n");
		exit(1);
	}
}

__global__ void generate_and_printf_random(curandStateMtgp32_t *state) {
	int a;
	a = curand(&state[blockIdx.x]);
	printf("(%d,%d):%d\n", blockIdx.x, threadIdx.x, a);
}

__global__ void generate_and_printf_random_float(curandStateMtgp32_t *state) {
	float a;
	a = curand_uniform(&state[blockIdx.x]);
	printf("(%d,%d):%f\n", blockIdx.x, threadIdx.x, a);
}


__global__ void generate_and_printf_random_specific(curandStateMtgp32_t *state) {
	int a;
	if(NUM_THREAD >= 2) {
		if(threadIdx.x == 0) {
			a = curand_mtgp32_specific(&state[blockIdx.x], 0, 2);
			printf("(%d,%d):%d\n", blockIdx.x, threadIdx.x, a);
		} else if(threadIdx.x == 1) {
			a = curand_mtgp32_specific(&state[blockIdx.x], 1, 2);
			printf("(%d,%d):%d\n", blockIdx.x, threadIdx.x, a);
		} else {
		}
	}
}

int main(void) {
	//デバイス上の変数・配列
	//\->シーケンス(パラメタの組)が入る配列
	mtgp32_kernel_params_t *dev_parameters_sequence;
	//\->乱数生成器の内部状態が入る配列
	//   curandStateMtgp32_t型変数一つの中に1024個の整数が入っていて、
	//   curandStateMtgp32_t型変数一つがMTGP32乱数生成器一つに対応する
	curandStateMtgp32_t *dev_generator_state;

	//ホスト上の変数・配列
	unsigned long long seed;
	seed = (unsigned long long)time(NULL);

	//配列を確保
	//\->ホスト
	//_\->今回は生成器の内部状態をブロック数の分だけ作る
	//     そして一ブロックあたり一つの生成器を使う
	check_cuda_funcs_executed_right (
		cudaMalloc((void **)&dev_generator_state, NUM_BLOCK * sizeof(curandStateMtgp32_t))
	);
	//_\->シーケンスを収める配列
	//    シーケンスは一つでいいの？
	check_cuda_funcs_executed_right (
		cudaMalloc((void **)&dev_parameters_sequence, sizeof(mtgp32_kernel_params_t))
	);

	//セットアップ
	//\->シーケンスをプリセットを使って設定
	check_curand_funcs_executed_right(
			curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, dev_parameters_sequence)
	);
	//\->乱数生成器の内部状態を設定
	//   今回は生成器を一ブロックあたり一つ用意するので、第四引数がNUM_BLOCKとなっている
	//   dev_generator_stateをセットするのにmtgp32dc_params_fast_11213を使ったのに
	//   ここでその両方を呼ぶのはなんでだろう
	check_curand_funcs_executed_right(
			curandMakeMTGP32KernelState(
				dev_generator_state, mtgp32dc_params_fast_11213,
				dev_parameters_sequence, NUM_BLOCK, seed
				)
	);
	//以上で、乱数生成器を使う準備ができた

	//すべてのスレッドでintの乱数を作って表示してみる
	generate_and_printf_random<<<NUM_BLOCK, NUM_THREAD>>>(dev_generator_state);
	cudaDeviceSynchronize();
	printf("\n");

	//一つのブロックのうち一部のスレッドでだけintの乱数を作って表示してみる
	//安全に行うために、curand_mtgp32_specific()を使う
	generate_and_printf_random_specific<<<NUM_BLOCK, NUM_THREAD>>>(dev_generator_state);
	cudaDeviceSynchronize();
	printf("\n");

	//すべてのスレッドでfloatの乱数を作って表示してみる
	generate_and_printf_random_float<<<NUM_BLOCK, NUM_THREAD>>>(dev_generator_state);
	cudaDeviceSynchronize();
	printf("\n");

	//配列を解放
	check_cuda_funcs_executed_right (
		cudaFree(dev_generator_state)
	);
	check_cuda_funcs_executed_right (
		cudaFree(dev_parameters_sequence)
	);
	return 0;
}

```

## 3.7. Thrustを使う例
## 3.8. 離散分布を作るコードの例
