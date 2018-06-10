#ライブラリの読み込み
require "nn"
require "nn/mnist"

#MNISTのトレーニング用データを読み込む
x_train, y_train = MNIST.load_train

#y_trainを10クラスに配列でカテゴライズする
y_train = MNIST.categorical(y_train)

#MNISTのテスト用データを読み込む
x_test, y_test = MNIST.load_test

#y_testを10クラスにカテゴライズする
y_test = MNIST.categorical(y_test)

puts "load mnist"

#ニューラルネットワークの初期化
nn = NN.new([784, 100, 100, 10], #ノード数
  learning_rate: 0.1, #学習率
  batch_size: 100, #ミニバッチの数
  activation: [:relu, :softmax], #活性化関数
  momentum: 0.9, #モーメンタム係数
  use_batch_norm: true, #バッチノーマライゼーションを使用する
)

#ミニバッチを0~1の範囲で正規化
func = -> x_batch, y_batch do
  x_batch /= 255
  [x_batch, y_batch]
end

#学習を行う
nn.train(x_train, y_train, 10, func) do
  #学習結果のテストを行う
  nn.test(x_test, y_test, &func)
end
