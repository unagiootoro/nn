require "nn"
require "nn/cifar10"

x_train = []
y_train = []

(1..5).each do |i|
  x_train2, y_train2 = CIFAR10.load_train(i)
  x_train.concat(x_train2)
  y_train.concat(CIFAR10.categorical(y_train2))
end
GC.start

x_test, y_test = CIFAR10.load_test
y_test = CIFAR10.categorical(y_test)
GC.start

puts "load cifar10"

nn = NN.new([3072, 100, 100, 10],
  learning_rate: 0.1,
  batch_size: 32,
  activation: [:relu, :softmax],
  momentum: 0.9,
  use_dropout: true,
  dropout_ratio: 0.2,
  use_batch_norm: true,
)

func = -> x, y do
  x /= 255
  [x, y]
end

nn.train(x_train, y_train, 50, func) do |epoch|
  nn.test(x_test, y_test, &func)
  nn.learning_rate *= 0.99
end
