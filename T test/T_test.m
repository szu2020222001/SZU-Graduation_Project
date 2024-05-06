close all
clear all
folder = 'G:\ZhangYuLin';
compare = 'HC_RPBD';
folder_hc = 'G:\ZhangYuLin\�����ؽ��\02_PBD_data';
folder_bd = 'G:\ZhangYuLin\�����ؽ��\03_RPBD_data';
subfolders = dir(folder_hc);
subfolderNames = {subfolders([subfolders.isdir]).name};  % ��ѡ���ļ���

% ȥ����ǰĿ¼��.�����ϼ�Ŀ¼��..��
subfolderNames = subfolderNames(~ismember(subfolderNames,{'.','..'}));

numSubfolders = numel(subfolderNames);

% fullPaths_hc = cell(1, numSubfolders);
% fullPaths_bd = cell(1, numSubfolders);
for x = 1:numSubfolders
    fullPaths_hc = fullfile(folder_hc, subfolderNames{x},strcat(subfolderNames{x}, '.mat'));
    fullPaths_bd = fullfile(folder_bd, subfolderNames{x},strcat(subfolderNames{x}, '.mat'));
    disp(fullPaths_hc);
    disp(fullPaths_bd);

    hc_data = load(fullPaths_hc);
    datavector0_h = hc_data.T_time;
    datavector0_h = permute(datavector0_h, [3, 1, 2]);  % [num,21,21]

    bd_data = load(fullPaths_bd);
    datavector0_q = bd_data.T_time;
    datavector0_q = permute(datavector0_q, [3, 1, 2]);  % [num,21,21]

    t=0.05;
    [h1,p1,s]=ttest2(datavector0_q,datavector0_h,t,'left'); %��ߣ��ұ�Ϊ1
    [h2,p2,s]=ttest2(datavector0_q,datavector0_h,t,'right'); %��ߣ��ұ�Ϊ1

    h1 = squeeze(h1);
    h2 = squeeze(h2);
    % ���� h1 - h2
    diff_h1_h2 = h1 - h2;

%     % ����ͼ��
%     figure;
%     imagesc(diff_h1_h2);
%     axis off;
%     colorbar;
%     caxis([-1, 1]);
%     title('h1 - h2');
    
    
    B1=squeeze(p1);
    B2=squeeze(p2); 
    p1=[];p2=[];
    for i=1:size(B1,1)
        for j=1:size(B1,2)
            if i==j
                p1(i,j)=0;
                p2(i,j)=0;
            else
                p1(i,j)=B1(i,j);
                p2(i,j)=B2(i,j);
            end
        end
    end
    p1= squeeze(p1);
    p2= squeeze(p2);
    % 
    % nvar = 21*21;
    % q = t./(nvar*(nvar-1));
    % 
    % p1(find(p1>q)) = 0;
    % p1(find(p1~=0)) = 1;
    % 
    % p2(find(p2>q)) = 0;
    % p2(find(p2~=0)) = 1;
    % bonferrion_result = p1-p2;

    % h1 =imagesc(bonferrion_result);
    % ���Ϊ1��ʾdatavector0_hǿ��datavector0_1��-1���ʾdatavector0_h����datavector0_q
    % axis off;

    [P_ida,P_idb] = fdr(p1,t);
    if isempty(P_ida)
        P_ida=0;
    end
    tem_p = p1;
    tem_p(tem_p>P_ida) = 0;
    FDR_p1 = tem_p;
    FDR_p1(FDR_p1 ~=0) = 1;

    [P_ida,P_idb] = fdr(p2,t);
    if isempty(P_ida)
        P_ida=0;
    end
    tem_p = p2;
    tem_p(tem_p>P_ida) = 0;
    FDR_p2 = tem_p;
    FDR_p2(FDR_p2 ~=0) = 1;

    FDR_P=FDR_p1-FDR_p2;% >0ԭʼ������ǿ��<0��ǿ�����ԭʼ
    FDR_P = squeeze(FDR_P);

    h_1 = imagesc(FDR_P');
    axis off;
    colorbar();
    caxis([-1,1]);

    figure('Position',[100 100 300 300]);
    
     % ���ñ���·�����ļ���
    fileName = strcat(subfolderNames{x}, '.png');
    fullPath = fullfile(folder, compare, fileName);

end
    
    
