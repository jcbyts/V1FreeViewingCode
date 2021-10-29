

%% Dot density calculation

num_dots = 1000;
frate = 240;
ppd = 37.5048;
screenSzDeg = [1280 720]/ppd;

ndeg = prod(screenSzDeg);

fprintf("%02.4f dots / deg^2 / frame\n", num_dots / ndeg)

%% Gabor density

num_gabors = 800;
ndeg = 20*20;

fprintf("%02.2f gabors / deg^2 / frame\n", num_gabors / ndeg)

%% DDPI system Precision
% artificial eye was rotated 1 degree on a rotational stage
D = load('Analysis/manuscript_freeviewingmethods/artificial_eye_1deg_09242019.mat');

gazex = D.dstruct.p4x-D.dstruct.p1x;
gazey = D.dstruct.p4y-D.dstruct.p1y;

valid_pre = 3000:11000;
valid_post = 14000:23000;

figure(1); clf
subplot(1,2,1)
plot(gazex); hold on
plot(valid_pre, gazex(valid_pre))
plot(valid_post,gazex(valid_post))
xlabel("Samples")
ylabel("raw measurement")

mu_pre = [mean(gazex(valid_pre)), mean(gazey(valid_pre))];
mu_post = [mean(gazex(valid_post)), mean(gazey(valid_post))];
mu_diff = mu_pre - mu_post;
onedeg = hypot(mu_diff(1), mu_diff(2));

x = [gazex(valid_pre) - mean(gazex(valid_pre)); gazex(valid_post) - mean(gazex(valid_post))];
y = [gazey(valid_pre) - mean(gazey(valid_pre)); gazey(valid_post) - mean(gazey(valid_post))];

x = x / onedeg;
y = y / onedeg;
z = x; % movement was along horizontal
nboot = 200;
ci = bootci(nboot, @rms, z);
m = rms(z);

subplot(1,2,2)
plot( (gazex(valid_pre(1):end) - mean(gazex(valid_pre))) / onedeg ); hold on
xlabel("samples")
ylabel("degrees")

fprintf('DDPI precision is: %02.4f [%02.4f, %02.4f] degrees\n', m, ci(1), ci(2))