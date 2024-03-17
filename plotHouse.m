% To plot images and points from mexVims and W
load HouseTallBasler64

for count = 1:NrFrames
  colormap gray
  axis equal
  image(bayer2rgb(mexVims(:,:,count)))
  hold on
  x = W(count,:);
  y = W(NrFrames+count,:);
  plot(x,y,'bo')
  title(sprintf('Frame %d', count));
  hold off
  drawnow;
  pause(0.1)
end

