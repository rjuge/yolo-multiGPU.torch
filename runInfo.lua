require 'paths'

local runInfo = {}

runInfo.fileName = "run.info"

function runInfo.writeToTxt(opt)
  
 file = io.open(paths.concat(opt.save, runInfo.fileName), "w")
 
 file:write(writeLine("Run:\t" .. opt.save))
 file:write(writeLine(""))

 file:write(writeLine("Seed:\t\t\t" .. opt.manualSeed))
 file:write(writeLine("Backend:\t\t" .. opt.backend))
 file:write(writeLine("Autotune:\t\t" .. opt.cudnnAutotune))
 file:write(writeLine("Primary GPU:\t\t" .. opt.GPU))
 file:write(writeLine("Number GPUs:\t\t" .. opt.nGPU))
 file:write(writeLine("Start From:\t\t" .. opt.retrain))
file:write(writeLine("Start From:\t\t" .. opt.optimState))
 file:write(writeLine(""))

  file:write(writeLine("Data:\t\t\t" .. opt.data))
  file:write(writeLine("Cache:\t\t\t" .. opt.cache))
  file:write(writeLine("Mapping File:\t\t" .. opt.classMapping))
  file:write(writeLine("Number Of Classes:\t" .. opt.nClasses))
  file:write(writeLine(""))
 
 file:write(writeLine("Architecture:\t\t" .. opt.netType))
 file:write(writeLine("Weight Init:\t\t" .. opt.wInit))
 file:write(writeLine("FT:\t\t\t" .. opt.FT))
 file:write(writeLine("Optimizer:\t\t" .. opt.optimizer))
 file:write(writeLine("Regime:\t\t\t" .. opt.regime))
 file:write(writeLine("LR:\t\t\t" .. opt.LR))
 file:write(writeLine("Momentum:\t\t" .. opt.momentum))
  file:write(writeLine("Weight Decay:\t\t" .. opt.weightDecay))
  file:write(writeLine(""))

file:write(writeLine("Max Epochs:\t\t" .. opt.nEpochs))
file:write(writeLine("Start Epoch:\t\t" .. opt.epochNumber))
  file:write(writeLine("Batch Size:\t\t" .. opt.batchSize))
  file:write(writeLine("Epoch Size:\t\t" .. opt.epochSize))
  file:write(writeLine(""))

  file:write(writeLine("Number Of Donkeys:\t\t" .. opt.nDonkeys))
  file:write(writeLine("Image Size:\t\t\t" .. opt.imageSize))
  file:write(writeLine("Crop Size:\t\t\t" .. opt.cropSize))
  file:write(writeLine("Proba Augmentation Train:\t" .. opt.PaugTrain))
  file:write(writeLine("Proba Augmentation Test:\t" .. opt.PaugTest))
  file:write(writeLine(""))
 
 file:close()

end

function writeLine(line)
  print(line)
  return line..'\n'
end


return runInfo

