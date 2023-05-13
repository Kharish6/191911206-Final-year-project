function count = featureExtraction(train_db,class_name,count)

files = dir([train_db '*.jpg']);
for i = 1:length(files)
    [dir_name] = files(i).name;
    
    files2 = [train_db dir_name];
    aa=imread(files2);
    [m,n,x]=size(aa);
    if x==3
    res_img = rgb2gray(imresize(aa,[100 100]));
    else
    res_img = (imresize(aa,[100 100]));
    end
    imm=imread(files2);
 
    X = double(res_img);
    [cA1,cH1,cV1,cD1] = dwt2(X,'haar');
    sx = size(X);
    A1 = idwt2(cA1,[],[],[],'haar',sx);  
    H1 = idwt2([],cH1,[],[],'haar',sx);
    V1 = idwt2([],[],cV1,[],'haar',sx);  
    D1 = idwt2([],[],[],cD1,'haar',sx);

    k = 7;
    [~,C] = kmeans(double(X),k);
    
    centers_val = mean(C,2);
    
    cooccur_matri = graycomatrix(res_img,'Offset',[2 0;0 2]);
    stats = glc_feature_analysis(cooccur_matri,0);
    energy = stats.energ;
    entrophy = stats.entro;
    contust = stats.contr;
    autoCorr = stats.autoc;
    prob = stats.maxpr;
    feat.val = [autoCorr ];
    feat.class = class_name;
    save(['Feature_Dir\feature_' num2str(count)],'feat');
    count = count+1;
end