-- Modified version of
-- https://github.com/Element-Research/rnn/blob/master/examples/sequence-to-one.lua

require 'rnn'
require 'os'
require 'cutorch'
require 'cunn'

-- hyper-parameters
batchSize = 128
rho = 64 -- sequence length
-- hiddenSize = 128
-- hiddenSize = 256
-- hiddenSize = 512
hiddenSize = 1024
nIndex = 10000-- input words
nClass = 5 -- output classes
lr = 0.1


-- build simple recurrent neural network
--r = nn.Recurrent(
   --hiddenSize, nn.Identity(),
   --nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   --rho
--)

r = nn.Sequential()
  :add(nn.FastLSTM(hiddenSize, hiddenSize))
  --:add(nn.LSTM(hiddenSize, hiddenSize))

rnn = nn.Sequential()
   :add(nn.LookupTable(nIndex, hiddenSize))
   :add(nn.SplitTable(1,2))
   :add(nn.Sequencer(r))
   :add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.LogSoftMax())

-- build criterion

criterion = nn.ClassNLLCriterion()

-- build dummy dataset (task is to predict class given rho words)
-- similar to sentiment analysis datasets
ds = {}
ds.size = 10000
ds.input = torch.LongTensor(ds.size,rho):cuda()
ds.target = torch.LongTensor(ds.size):random(nClass):cuda()

-- this will make the inputs somewhat correlate with the targets,
-- such that the reduction in training error should be more obvious
local correlate = torch.LongTensor(nClass, rho*3):random(nClass)
local indices = torch.LongTensor(rho)
local buffer = torch.LongTensor()
local sortVal, sortIdx = torch.LongTensor(), torch.LongTensor()
for i=1,ds.size do
   indices:random(1,rho*3)
   buffer:index(correlate[ds.target[i]], 1, indices)
   sortVal:sort(sortIdx, buffer, 1)
   ds.input[i]:copy(sortVal:view(-1))
end



indices:resize(batchSize)

inputs, targets = torch.LongTensor():cuda(), torch.LongTensor():cuda()

print(inputs, targets)

rnn:cuda()
indices:cuda()

inputs:cuda()
targets:cuda()
print(inputs, targets)

criterion:cuda()
ds.input:cuda()
ds.target:cuda()

-- 1. create a sequence of rho time-steps

indices:random(1, ds.size) -- choose some random samples
inputs:index(ds.input, 1,indices)
targets:index(ds.target, 1, indices)

function step()
   -- 2. forward sequence through rnn
   rnn:zeroGradParameters()
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward(inputs, gradOutputs)
   -- 4. update
   rnn:updateParameters(lr)
end

iteration = 0
step()
step()

t_start = os.clock()
-- training
for iteration = 1, 50 do
  step()
end

t_diff = os.clock() - t_start
print(string.format("Took %f sec", t_diff))
print(t_diff / 50.)
