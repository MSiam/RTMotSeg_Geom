require 'nngraph'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
require 'ResizeJoinTable'

local threads = require 'threads'
threads.serialization('threads.sharedserialize')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')

local params = cmd:parse(arg)
cutorch.setDevice(params.gpu + 1)

math.randomseed(os.time())
local seed = math.random()

local pool = threads.Threads(
    3,
    function()
        require 'torch'
        require 'cunn'
        require 'cutorch'
        require 'form_batch_angle'
        cutorch.setDevice(params.gpu + 1)
    end,
    function(threadid)
        math.randomseed(seed + threadid)
        print('starting a new thread/state number ' .. threadid)
        trainList = load_dataset('training_mpnet.txt')
    end
)

local modelName = 'model'
print('Traininig ' .. modelName)

dofile('model.lua')

mlp:float()
mlp:training()
mlp:cuda()

local params, gradParams = mlp:getParameters()
local criterion = nn.BCECriterion()
criterion = criterion:float()
criterion = criterion:cuda()
local trainSize = 1300;
local batchSize = 1;
local lossN = 10;
local numIter = torch.round(trainSize / batchSize)

local optimState = {learningRate=0.003,momentum=0.9,weightDecay=0.005,dampening = 0.0}

local iter = 0
local lossEpoche = 0
local epocheOver = false
for epoche = 1, 27 do
    print('Starting Epoche ' .. epoche)
    if epoche == 10 then
        optimState.learningRate = 0.0003
        optimState.weightDecay=0.0005
    end
    if epoche == 19 then
        optimState.learningRate = 0.00003
        optimState.weightDecay=0.00005
    end

    while true do

        pool:addjob(
        -- the job callback (runs in data-worker thread)
            function()
                collectgarbage()
                local batch_ims, batch_labels = get_batch(batchSize)
                batch_ims = batch_ims:float():cuda()
                batch_labels = batch_labels:float():cuda()
                return batch_ims, batch_labels
            end,
            -- the end callback (runs in the main thread)
            function (batch_ims, batch_labels)
                local function feval(params)
                    collectgarbage()

                    gradParams:zero()

                    local outputs = mlp:forward(batch_ims)

                    local loss = criterion:forward(outputs, batch_labels)
                    lossEpoche = lossEpoche - lossEpoche / lossN
                    lossEpoche = lossEpoche + loss / lossN
                    local dloss_doutput = criterion:backward(outputs, batch_labels)
                    mlp:backward(batch_ims, dloss_doutput)

                    return loss,gradParams
                end

                optim.sgd(feval, params, optimState)
                iter = iter + 1

                if iter == numIter then
                    epocheOver = true
                end
            end
        )

        if epocheOver then
            iter = 0
            epocheOver = false
            break
        end
    end

    print('Epoche ' .. epoche .. ' finished')
    print('Training loss: ' .. lossEpoche)
    lossEpoche = 0
end

pool:synchronize()

mlp:clearState()
torch.save('models/' .. modelName .. '.dat', mlp)

