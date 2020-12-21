clear 
close all
clc

load pointcloud.mat
myvals = whos;

fig = figure(1);
fig.Position(3) = fig.Position(3)*length(myvals);
fig.Position(4) = fig.Position(4)*2;
hold on

for n = 1:length(myvals)
    
    figure(1);
    subplot(2, length(myvals), n);
        
    temp = eval(myvals(n).name);
    
    pc = temp(1:3, :);
    rgb = uint8( temp(4:6, :) );
      
    pcshow( pointCloud(pc','color', rgb') );
    % saveas(img, ['Point Cloud: ', myvals(n).name]) % é mais facil
    % escolher o melhor angulo à pata

    
    title(['Point Cloud: ', myvals(n).name]);
    xlabel('X label');
    ylabel('Y label');
    zlabel('Z label');
    
    subplot(2, length(myvals), n+2);
    title(['RGB frame: ', myvals(n).name]);
    new_image = reshape(rgb', 640, 480, 3);
    imagesc(permute(new_image, [2,1,3]));
    
    figure(2);
    img = imagesc(permute(new_image, [2,1,3]));
    saveas(img, "RGB frame: " + myvals(n).name + ".png")
    close 2;
    

end

