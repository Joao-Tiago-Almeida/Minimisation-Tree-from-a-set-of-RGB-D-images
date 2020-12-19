clear 
close all
clc

load pointcloud.mat
myvals = whos;

fig = figure();
fig.Position(3) = fig.Position(3)*length(myvals);
hold on

for n = 1:length(myvals)
    
    subplot(1, length(myvals),n);
        
    pc = eval(myvals(n).name)';
      
    pcshow(pc);
    
    title(['Point Cloud ', num2str(n)]);
    xlabel('X label');
    ylabel('Y label');
    zlabel('Z label');

end