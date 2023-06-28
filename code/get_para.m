X = cat(2, load("../dataset/msf.mat").msf, load("../dataset/pan.mat").pan);
X = double(X);
[sj, para, val] = icanfast(X)