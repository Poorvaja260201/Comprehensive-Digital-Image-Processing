% Load color image
img = imread('C:\Users\Poorvaja\Downloads\flower.jpg'); % Replace with your image file
gray_img = rgb2gray(img);

% Display grayscale image
figure, imshow(gray_img), title('Grayscale Image');
% Calculate histogram
hist_values = imhist(gray_img);

% Plot histogram
figure, bar(0:255, hist_values), title('Histogram');
xlabel('Pixel Intensity'), ylabel('Frequency');
mean_val = mean(gray_img(:));
variance_val = var(double(gray_img(:)));
std_dev = std(double(gray_img(:)));

fprintf('Mean: %.2f\n', mean_val);
fprintf('Variance: %.2f\n', variance_val);
fprintf('Standard Deviation: %.2f\n', std_dev);
img="C:\Users\Poorvaja\Downloads\flower.jpg"

% Gamma correction function
gamma_correct = @(img, gamma) uint8(255 * ((double(img) / 255) .^ gamma));

% Apply gamma correction
gamma_0_5 = gamma_correct(gray_img, 0.5);
gamma_2_0 = gamma_correct(gray_img, 2.0);

% Display results
figure;
subplot(1, 2, 1), imshow(gamma_0_5), title('Gamma 0.5');
subplot(1, 2, 2), imshow(gamma_2_0), title('Gamma 2.0');
% Perform histogram equalization
equalized_img = histeq(gray_img);

% Display the result
figure, imshow(equalized_img), title('Histogram Equalized Image');
% Apply Otsu's thresholding
threshold = graythresh(gray_img); % Otsu's threshold (normalized)
otsu_thresh_img = imbinarize(gray_img, threshold);

% Display the result
figure, imshow(otsu_thresh_img), title('Otsu Thresholding');
% Adaptive thresholding using local mean
window_size = 11; % Adjust window size if needed
adaptive_thresh_img = adaptthresh(gray_img, 'NeighborhoodSize', window_size, 'ForegroundPolarity', 'dark');
adaptive_binary_img = imbinarize(adaptive_thresh_img);

% Display the result
figure, imshow(adaptive_binary_img), title('Adaptive Thresholding');

% Add Gaussian noise with standard deviation 0.01 and 0.05
noisy_gauss_001 = imnoise(gray_img, 'gaussian', 0, 0.01);
noisy_gauss_005 = imnoise(gray_img, 'gaussian', 0, 0.05);

% Display noisy images
figure;
subplot(1, 2, 1), imshow(noisy_gauss_001), title('Gaussian Noise (σ = 0.01)');
subplot(1, 2, 2), imshow(noisy_gauss_005), title('Gaussian Noise (σ = 0.05)');
% Add salt-and-pepper noise with density 0.02
noisy_snp = imnoise(gray_img, 'salt & pepper', 0.02);

% Display noisy image
figure, imshow(noisy_snp), title('Salt-and-Pepper Noise (d = 0.02)');
% Function to calculate SNR
function snr_val = calculate_snr(original, noisy)
    original = double(original);
    noisy = double(noisy);
    signal_power = sum(original(:).^2) / numel(original);
    noise_power = sum((original(:) - noisy(:)).^2) / numel(noisy);
    snr_val = 10 * log10(signal_power / noise_power);
end

% Compute SNR for noisy images
snr_gauss_001 = calculate_snr(gray_img, noisy_gauss_001);
snr_gauss_005 = calculate_snr(gray_img, noisy_gauss_005);
snr_snp = calculate_snr(gray_img, noisy_snp);

fprintf('SNR (Gaussian σ=0.01): %.2f dB\n', snr_gauss_001);
fprintf('SNR (Gaussian σ=0.05): %.2f dB\n', snr_gauss_005);
fprintf('SNR (Salt-and-Pepper d=0.02): %.2f dB\n', snr_snp);
% Apply Gaussian filter
gaussian_filter = fspecial('gaussian', [5, 5], 1); % Size 5x5, σ=1
gaussian_filtered = imfilter(noisy_gauss_005, gaussian_filter, 'replicate');

% Display result
figure, imshow(gaussian_filtered), title('Gaussian Filter');
% Apply median filtering
median_filtered = medfilt2(noisy_snp, [3, 3]);

% Display result
figure, imshow(median_filtered), title('Median Filter');
% PSNR for median filtering
psnr_median = psnr(median_filtered, gray_img);

fprintf('PSNR (Median Filter): %.2f dB\n', psnr_median);
% Sobel edge detection
edges_sobel = edge(gray_img, 'sobel');

% Display result
figure, imshow(edges_sobel), title('Sobel Edge Detection');
% Prewitt edge detection
edges_prewitt = edge(gray_img, 'prewitt');

% Display result
figure, imshow(edges_prewitt), title('Prewitt Edge Detection');
% Display side-by-side comparison
figure;
subplot(1, 2, 1), imshow(edges_sobel), title('Sobel Edge Detection');
subplot(1, 2, 2), imshow(edges_prewitt), title('Prewitt Edge Detection');
% Canny edge detection
edges_canny = edge(gray_img, 'canny');

% Display result
figure, imshow(edges_canny), title('Canny Edge Detection');
% Display comparison between Sobel and Canny
figure;
subplot(1, 2, 1), imshow(edges_sobel), title('Sobel Edge Detection');
subplot(1, 2, 2), imshow(edges_canny), title('Canny Edge Detection');


% Display all methods side-by-side
figure;
subplot(2, 2, 1), imshow(edges_sobel), title('Sobel');
subplot(2, 2, 2), imshow(edges_prewitt), title('Prewitt');
subplot(2, 2, 3), imshow(edges_canny), title('Canny');
subplot(2, 2, 4), imshow(edges_sobel), title('Sobel (Repeated)');
binary_img = imbinarize(gray_img); % Convert grayscale to binary using Otsu's method
figure, imshow(binary_img), title('Binary Image');
% Create a structuring element
se = strel('disk', 3); % Disk-shaped structuring element with radius 3

% Dilation
dilated_img = imdilate(binary_img, se);
figure, imshow(dilated_img), title('Dilated Image');

% Erosion
eroded_img = imerode(binary_img, se);
figure, imshow(eroded_img), title('Eroded Image');
opened_img = imopen(binary_img, se);
figure, imshow(opened_img), title('Opened Image');
closed_img = imclose(binary_img, se);
figure, imshow(closed_img), title('Closed Image');
boundary = binary_img - imerode(binary_img, se);
figure, imshow(boundary), title('Boundary Extraction');
filled_img = imfill(binary_img, 'holes');
figure, imshow(filled_img), title('Hole-Filled Image');

% Read and reshape the image
img = imread('C:\Users\Poorvaja\Downloads\flower.jpg');
img_resized = imresize(img, [256 256]); % Resize for faster processing
img_data = double(reshape(img_resized, [], 3)); % Reshape to N x 3 (RGB format)

% Test different k values
k_values = [2, 4, 6]; % Cluster counts
segmented_images = cell(size(k_values));

for i = 1:length(k_values)
    k = k_values(i);
    % Apply k-means clustering
    [cluster_idx, cluster_centers] = kmeans(img_data, k, 'MaxIter', 200);
    
    % Assign each pixel the RGB value of its cluster center
    clustered_img = reshape(cluster_centers(cluster_idx, :), size(img_resized));
    segmented_images{i} = uint8(clustered_img);

    % Display segmented image
    figure, imshow(segmented_images{i}), title(['K-means Segmentation (k = ', num2str(k), ')']);
end


% Add Mean Shift function (from a library or toolbox)
addpath('path_to_mean_shift_library');

% Define parameters
bandwidth = [10, 20, 30]; % Test different bandwidth values

for b = 1:length(bandwidth)
    % Apply mean shift
    segmented_img = mean_shift_segmentation(img_resized, bandwidth(b));

    % Display segmented image
    figure, imshow(segmented_img), title(['Mean Shift Segmentation (Bandwidth = ', num2str(bandwidth(b)), ')']);
end
smoothed_img = imgaussfilt(img_resized, 2); % Simulate bandwidth smoothing
[cluster_idx, cluster_centers] = kmeans(double(reshape(smoothed_img, [], 3)), 5);

% Reshape back to image
clustered_img = reshape(cluster_centers(cluster_idx, :), size(img_resized));
figure, imshow(uint8(clustered_img)), title('Simulated Mean Shift Segmentation');
function [gaussian_pyramid] = gaussianPyramid(img, levels)
    gaussian_pyramid = cell(levels, 1);  % Initialize pyramid with the specified number of levels
    gaussian_pyramid{1} = img;  % First level is the original image

    for i = 2:levels
        blurred_img = imgaussfilt(gaussian_pyramid{i-1}, 1);  % Apply Gaussian filter
        downsampled_img = imresize(blurred_img, 0.5);  % Downsample by factor of 2
        gaussian_pyramid{i} = downsampled_img;
    end
function [laplacian_pyramid] = laplacianPyramid(gaussian_pyramid)
    levels = length(gaussian_pyramid);
    laplacian_pyramid = cell(levels-1, 1); % One less level than Gaussian pyramid
    
    for i = 1:levels-1
        % Upsample and blur the next level
        upsampled_img = imresize(gaussian_pyramid{i+1}, size(gaussian_pyramid{i}(:,:,1)));
        blurred_upsampled_img = imgaussfilt(upsampled_img, 1);
        
        % Subtract to get Laplacian level
        laplacian_pyramid{i} = gaussian_pyramid{i} - blurred_upsampled_img;
    end
end
% Load an image
img = imread('image.jpg');
gray_img = rgb2gray(imresize(img, [256 256])); % Convert to grayscale and resize

% Create Gaussian and Laplacian pyramids
gaussian_pyramid = gaussianPyramid(gray_img, 4);
laplacian_pyramid = laplacianPyramid(gaussian_pyramid);

% Display Gaussian Pyramid
figure;
for i = 1:length(gaussian_pyramid)
    subplot(1, 4, i), imshow(gaussian_pyramid{i}, []), title(['Gaussian Level ', num2str(i)]);
end

% Display Laplacian Pyramid
figure;
for i = 1:length(laplacian_pyramid)
    subplot(1, 3, i), imshow(laplacian_pyramid{i}, []), title(['Laplacian Level ', num2str(i)]);
end
% Load an image
img = imread('image.jpg');
gray_img = rgb2gray(imresize(img, [256 256])); % Convert to grayscale and resize

% Create Gaussian and Laplacian pyramids
gaussian_pyramid = gaussianPyramid(gray_img, 4);
laplacian_pyramid = laplacianPyramid(gaussian_pyramid);

% Display Gaussian Pyramid
figure;
for i = 1:length(gaussian_pyramid)
    subplot(1, 4, i), imshow(gaussian_pyramid{i}, []), title(['Gaussian Level ', num2str(i)]);
end

% Display Laplacian Pyramid
figure;
for i = 1:length(laplacian_pyramid)
    subplot(1, 3, i), imshow(laplacian_pyramid{i}, []), title(['Laplacian Level ', num2str(i)]);
end
function [bilateral_img] = bilateralFilter(img, spatial_sigma, intensity_sigma)
    % Apply bilateral filter
    bilateral_img = imguidedfilter(img, 'DegreeOfSmoothing', intensity_sigma, ...
                                   'NeighborhoodSize', spatial_sigma);
end



% Load the image
img = imread('image.jpg');
gray_img = rgb2gray(imresize(img, [256 256])); % Convert to grayscale and resize

% Parameters for bilateral filter
spatial_sigma = 5; % Spatial kernel size
intensity_sigma = 0.1; % Intensity similarity threshold

% Apply bilateral filter
bilateral_img = bilateralFilter(gray_img, spatial_sigma, intensity_sigma);

% Display results
figure;
subplot(1, 2, 1), imshow(gray_img), title('Original Image');
subplot(1, 2, 2), imshow(bilateral_img), title('Bilateral Filtered Image');





