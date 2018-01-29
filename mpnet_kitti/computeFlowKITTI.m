function computeFlowDAVIS()  
    davisPath = '/usr/data/Datasets/KITTI_MOD/';
    addpath(genpath('.'))
%    seqs = dir([davisPath, '/testing_1/images/']);
%    for i = 3 : length(seqs)
%        seqs(i).name
    computeFlowSeq(davisPath);
%    end        
end

function computeFlowSeq(davisPath)
    [davisPath, 'testing/images/']
    frames = dir([davisPath, 'testing/images/']);
    mkdir([davisPath, 'testing/MPNET_OpticalFlow/'])
    fid = fopen([davisPath, 'testing/MPNET_OpticalFlow/minmax.txt'], 'w');
    for i = 16 : length(frames) - 1
        frame_name = frames(i).name;
        split = strsplit(frame_name, '.');
        frame1 = imread([davisPath, 'testing/images/'...
            , frames(i).name]);
        frame2 = imread([davisPath, 'testing/images/' ...
            , frames(i + 1).name]);
        
        [flow, ~] = sundaramECCV10_ldof_GPU_mex(frame1, frame2);
        
        temp = flow(:, :, 2);
        flow(:, :, 2) = flow(:, :, 1);        
        flow(:, :, 1) = temp;
                
        baseVector = zeros(size(flow, 1), size(flow, 2), 2);        
        baseVector(:, :, 1) = 1;
        
        angleField = acos(dot(flow, baseVector, 3) ./ ...
            ((sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2)) .* ...
            ones(size(flow, 1), size(flow, 2))));
        magnitudes = sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2);
        
        minAngle = min(angleField(:));
        maxAngle = max(angleField(:));
        angleField = (angleField - minAngle) ./ (maxAngle - minAngle);
        
        imwrite(angleField, [davisPath, '/testing/MPNET_OpticalFlow/' ...
            , '/angleField_', split{1}, '.jpg']);
        
        minMagnitude = min(magnitudes(:));
        maxMagnitude = max(magnitudes(:));
        magnitudes = (magnitudes - minMagnitude) ./ (maxMagnitude - minMagnitude);
        
        imwrite(magnitudes, [davisPath, '/testing/MPNET_OpticalFlow/' ...
            , '/magField_', split{1}, '.jpg']);
        
        fprintf(fid, '%f %f %f %f\n', minAngle, maxAngle, minMagnitude, maxMagnitude);
    end

    fclose(fid);    
end
