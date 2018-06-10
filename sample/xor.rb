#ライブラリの読み込み
require "nn"

x = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
]

y = [[0], [1], [1], [0]]

#ニューラルネットワークの初期化
nn = NN.new([2, 4, 1], #ノード数
  learning_rate: 0.1, #学習率
  batch_size: 4, #ミニバッチの数
  activation: [:sigmoid, :identity] #活性化関数
)

#学習を行う
nn.train(x, y, 20000)

#学習結果の確認
p nn.run(x)
