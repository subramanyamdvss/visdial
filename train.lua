require 'nn'
require 'nngraph'
require 'rnn'

------------------------------------------------------------------------
-- Input arguments and options
------------------------------------------------------------------------
local opt = require 'opts';
print(opt)

-- seed for reproducibility
torch.manualSeed(1234);

------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, {'train'});
collectgarbage();

-- set default tensor based on gpu usage
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.setDevice(opt.gpuid+1)
    cutorch.manualSeed(1234)
    torch.setdefaulttensortype('torch.CudaTensor');
else
    torch.setdefaulttensortype('torch.FloatTensor');
end

------------------------------------------------------------------------
-- Setting model parameters
------------------------------------------------------------------------
-- transfer all options to model
local modelParams = opt;

-- transfer parameters from dataloader to model
paramNames = {'numTrainThreads', 'numTestThreads', 'numValThreads',
                'vocabSize', 'maxQuesCount', 'maxQuesLen', 'maxAnsLen'};
for _, value in pairs(paramNames) do
    modelParams[value] = dataloader[value];
end

-- path to save the model
local modelPath = opt.savePath

-- creating the directory to save the model
paths.mkdir(modelPath);

-- Iterations per epoch
modelParams.numIterPerEpoch = math.ceil(modelParams.numTrainThreads /
                                                modelParams.batchSize);
print(string.format('\n%d iter per epoch.', modelParams.numIterPerEpoch));

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'model'
local model = Model(modelParams);

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------

function topk(list, k, comp)
    collectgarbage()
    cutorch.synchronize()
    time = sys.clock()
    local subset = (list.data)
    print(torch.type(subset))
    torch.setdefaulttensortype('torch.CudaTensor' )
    for i = 1,(list:size()),(opt.batchSize*40) do
        xlua.progress(i,list:size())
        local btch = subset:sub(i,math.min(i+opt.batchSize*40-1,list:size()))
        if i == 1 then
            logprob = net:forward(btch)
        else
        logprob = torch.cat({logprob,net:forward(btch)},1)
        end
    end
    cutorch.synchronize()
    print('\ntime taken',sys.clock()-time)
    time = sys.clock()
    collectgarbage()
    subset = -((torch.cmul(logprob,torch.exp(logprob))*(torch.Tensor(#classes,1):fill(1))):t())
    print(torch.type(subset))
    y , list2 = torch.sort(subset,2,true)
    cutorch.synchronize()
    print('time taken',sys.clock()-time)
    time = sys.clock()
    print(torch.type(list2),torch.type(list.data),torch.type(list.label))
    list.data:indexCopy(1,list2[1],(list.data))
    list.label:indexCopy(1,list2[1],(list.label))
    cutorch.synchronize()
    print('time taken',sys.clock()-time)
    time = sys.clock()
    list2:free()
    batch = list.data:sub(1,k)
    labels = list.label:sub(1,k)
    collectgarbage()
    list.data = list.data:sub(k+1,(#(list.data))[1]) 
    collectgarbage()
    list.label = list.label:sub(k+1,(#(list.label))[1]) 
    collectgarbage()
    print('dvss')
    cutorch.synchronize()
    print('time taken',sys.clock()-time)
    time = sys.clock()
    return batch,labels,list;
end 


print('Training..')
collectgarbage()

runningLoss = 0;
for iter = 1, modelParams.numEpochs * modelParams.numIterPerEpoch do
    -- forward and backward propagation
    model:trainIteration(dataloader);
    
    -- evaluate on val and save model
    if iter % (5 * modelParams.numIterPerEpoch) == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch
        
        -- save model and optimization parameters
        torch.save(string.format(modelPath .. 'model_epoch_%d.t7', currentEpoch),
                                                    {modelW = model.wrapperW,
                                                    optims = model.optims,
                                                    modelParams = modelParams})
        -- validation accuracy
        -- model:retrieve(dataloader, 'val');
    end
    
    -- print after every few iterations
    if iter % 100 == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch;
        
        -- print current time, running average, learning rate, iteration, epoch
        print(string.format('[%s][Epoch:%.02f][Iter:%d][Loss:%.05f][lr:%f]',
                                os.date(), currentEpoch, iter, runningLoss, 
                                            model.optims.learningRate))
    end
    if iter % 10 == 0 then collectgarbage(); end
end

-- Saving the final model
torch.save(modelPath .. 'model_final.t7', {modelW = model.wrapperW:float(),
                                            modelParams = modelParams});
