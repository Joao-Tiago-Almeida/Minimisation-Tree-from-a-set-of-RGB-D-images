data = load('allpointcloud.mat')

figure(1);
hold on
plot3(data.world(1,:,:), data.world(2,:,:), data.world(3,:,:), '.')
plot3(data.two(1,:,:), data.two(2,:,:), data.two(3,:,:), '.')
legend({'World prespective';'RGB-D image 3'})
title('3D points in world prespective', 'FontSize', 20);
xlabel('X label');
ylabel('Y label');
zlabel('Z label');
