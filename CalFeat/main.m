% 指定主文件夹路径
mainFolderPath = 'D:\PycharmProjects\DeepLearning\Graduation Project\v0313\Data Analyse\Processed raw data_240313';
% 获取主文件夹中的所有文件夹信息
mainFolderList = dir(mainFolderPath);

% 创建进度条
totalFolders = sum([mainFolderList.isdir]) - 2; % 总文件夹数（减去'.'和'..'）
progressBar = waitbar(0, 'Processing Files...');

% 循环遍历主文件夹下的所有文件夹
for i = 1:length(mainFolderList)
    % 排除当前目录（'.'）和上一级目录（'..'）
    if ~strcmp(mainFolderList(i).name, '.') && ~strcmp(mainFolderList(i).name, '..') && mainFolderList(i).isdir
        % 获取子文件夹的名称
        subFolderName = mainFolderList(i).name;
        
        % 构建子文件夹的完整路径
        subFolderPath = fullfile(mainFolderPath, subFolderName);
        
        % 使用dir函数获取子文件夹中的所有子文件夹
        subSubFolderList = dir(subFolderPath);
        
        % 循环遍历子文件夹中的所有子文件夹
        for j = 1:length(subSubFolderList)
            % 排除当前目录（'.'）和上一级目录（'..'）
            if ~strcmp(subSubFolderList(j).name, '.') && ~strcmp(subSubFolderList(j).name, '..') && subSubFolderList(j).isdir
                % 获取当前子文件夹的名称
                subSubFolderName = subSubFolderList(j).name;
                
                % 构建当前子文件夹的完整路径
                subSubFolderPath = fullfile(subFolderPath, subSubFolderName);
                
                % 使用dir函数获取当前子文件夹中的所有mat文件
                matFileList = dir(fullfile(subSubFolderPath, '*.mat'));
                
                % 循环遍历当前子文件夹中的所有mat文件
                for k = 1:length(matFileList)
                    % 获取mat文件名
                    matFileName = matFileList(k).name;
                    
                    % 获取mat文件的完整路径
                    matFilePath = fullfile(subSubFolderPath, matFileName);
                    
                    % 在这里进行你的操作，比如读取mat文件内容等
                    fprintf('正在处理文件：%s\n', matFileName);
                    
                    % 读取mat文件内容
                    load(matFilePath, 'T_time'); % 假设mat文件中有变量T_time
                    
                    % 进行操作
                    for sub = 1:size(T_time,3)
                        Re_CorrMatrix = squeeze(T_time(:,:,sub)) + squeeze(T_time(:,:,sub))';
                        [m,n]=size(Re_CorrMatrix);
                        %聚类系数
                        Propertise(1,:,sub) = clustering_coef_wu(Re_CorrMatrix);
                        %局部效率
                        Propertise(2,:,sub) = efficiency_wei(Re_CorrMatrix,1);
                        % 权重度
                        for j = 1:n
                            Propertise(3,j,sub) = sum(Re_CorrMatrix(j,:));
                        end
                        % closeness centrality
                        Matrix11 = ones(size(Re_CorrMatrix))-Re_CorrMatrix;
                        Matrix11(1:n+1:end) = 0;
                        [D11 B11] = distance_wei(Matrix11);%D11为两个节点的最短路径长度
                        for j = 1:n
                            Propertise(4,j,sub) = sum(D11(j,:));
                        end
                        % 特征向量中心性
                        Propertise(5,:,sub)=eigenvector_centrality_und(Re_CorrMatrix);
                        
                        % 保存结果
                        [~, matFileNameWithoutExt, ~] = fileparts(matFileName);
                        propertise_result = fullfile(subSubFolderPath, [matFileNameWithoutExt, '_Propertise.mat']);
                        save(propertise_result, 'Propertise');
                        
                        clear Re_CorrMatrix Matrix11
                    end
                end
            end
        end
        
        % 更新进度条
        waitbar(i / totalFolders, progressBar, sprintf('Processing Files... %d/%d', i, totalFolders));
    end
end

% 关闭进度条
close(progressBar);
