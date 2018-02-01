require 'lfs'
require 'image'

function load_dataset(frameListFile)
    local file = io.open("/home/eren/Data/KITTI_MOD/training/" .. frameListFile)

    local trainList = {}
    if file then
        for line in file:lines() do
            table.insert(trainList, line)
        end
    else
        print('File not found')
    end

    order = torch.randperm(#trainList)

    return trainList
end

count = 1;
order = 0;

function get_batch(batch_size)
    local batch_ims = torch.Tensor(batch_size, 2, 200, 360)
    --local batch_ims = torch.Tensor(batch_size, 2, 375, 1242)
    local batch_labels = torch.Tensor(batch_size, 1, 100, 180)
    --local batch_labels = torch.Tensor(batch_size, 1, 187, 621 )
    main_dir= '/home/eren/Data/KITTI_MOD/training/'
    for i = 1, batch_size do
        local flowPath, path;
        --local f = io.open(trainList[count], "r")
        --print('element '.. trainList[count])
        --        repeat
        --print(path)
        path = main_dir .. trainList[order[count]]
        flowPath = string.gsub(path, 'old_mask', 'MPNET_OpticalFlow');
        flowPath = string.gsub(flowPath, 'png', 'jpg');
        --            local flowAnglePath = string.gsub(flowPath, 'MPNet_OpticalFlow/', 'MPNet_OpticalFlow/angleField_')
        --
        --            if f then
        --                io.close(f)
        --            end
        --            f = io.open(flowAnglePath, "r")
        --
        --            count = count + 1;
        --            if count > #trainList then
        --                count = 1
        --                order = torch.randperm(#trainList)
        --            end
        --        until f
        --io.close(f)

        local labels = image.load(path)
        local flowAnglePath = string.gsub(flowPath, 'MPNET_OpticalFlow/', 'MPNET_OpticalFlow/angleField_')
        local flowAngle = image.load(flowAnglePath)
        local flowMagPath = string.gsub(flowPath, 'MPNET_OpticalFlow/', 'MPNET_OpticalFlow/magField_')
        local flowMag = image.load(flowMagPath)

        local flow = torch.cat(flowAngle, flowMag, 1)

        local minMaxPath = '/home/eren/Data/KITTI_MOD/training/MPNET_OpticalFlow/minmax.txt'--string.gsub(flowPath, '%d+.jpg', 'minmax.txt')

        local file = io.open(minMaxPath)
        local minmaxes = {}
        local ind = 1
        if file then
            for line in file:lines() do
                local min_x, max_x, min_y, max_y = unpack(line:split(" "))
                minmaxes[ind]  = {min_x, max_x, min_y, max_y}
                ind = ind + 1
            end
        else
            print('File not found!!!!!!!!')
        end
        io.close(file)

        local frameIndStart, frameIndEnd = string.find(path, '%d+.png')
        local frameInd = tonumber(path:sub(frameIndStart, frameIndEnd - 4)) - 5

        local mm = minmaxes[frameInd]
        --if not mm then
        --    print(path)
        --end

        flow = image.scale(flow, 480, 270, 'simple')
        labels = image.scale(labels, 480, 270, 'simple')
        --flow[{{1}, {}, {}}] = flow[{{1}, {}, {}}] * (mm[2] - mm[1]) + mm[1]
        --flow[{{2}, {}, {}}] = flow[{{2}, {}, {}}] * (mm[4] - mm[3]) + mm[3]

        flow[{{2}, {}, {}}]:div(math.sqrt(math.pow(960, 2) + math.pow(540, 2)) / 6)


        local h1 = math.ceil(torch.uniform(1e-2, 270 - 200))
        local w1 = math.ceil(torch.uniform(1e-2, 480 - 360))
        flow = flow[{{}, {h1, h1 + 200 - 1}, {w1, w1 + 360 - 1}}]
        labels = labels[{{}, {h1, h1 + 200 - 1}, {w1, w1 + 360 - 1}}]

        if math.random() > 0.5 then
            flow = image.flip(flow:contiguous(), 3);
            labels = image.flip(labels:contiguous(), 3);
        end

        labels = image.scale(labels, 180, 100, 'simple')
        --labels = image.scale(labels, 621, 187, 'simple')

        batch_ims[i] = flow
        batch_labels[i] = labels
    end

    return batch_ims, batch_labels

end







