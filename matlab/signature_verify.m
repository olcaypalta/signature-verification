function results = signature_verify(refDir, queryDir, k)
%SIGNATURE_VERIFY Offline signature verification with a few genuine references.
%   results = signature_verify('../examples/genuine', '../examples/questioned')
%   results = signature_verify(refDir, queryDir, 2.5)
%
%   Features: HOG + LBP (4x8 grid) + global shape descriptors.
%   Decision: mean cosine distance to references, accepted if
%             (d - mu) / sigma <= k, with mu/sigma from pairwise reference distances.
%   Requires: Computer Vision Toolbox, Image Processing Toolbox.

if nargin < 1, refDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'examples', 'genuine'); end
if nargin < 2, queryDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'examples', 'questioned'); end
if nargin < 3, k = 2.0; end

refs = list_images(refDir);
if numel(refs) < 3, error('need at least 3 reference signatures'); end
R = cell2mat(cellfun(@sig_features, refs, 'UniformOutput', false));

n = size(R, 1); pair = [];
for i = 1:n
    for j = i+1:n
        pair(end+1) = cos_dist(R(i, :), R(j, :)); %#ok<AGROW>
    end
end
mu = mean(pair); sigma = std(pair) + 1e-4;
fprintf('enrolled %d references: intra distance mu=%.4f sigma=%.4f\n', n, mu, sigma);

queries = list_images(queryDir);
results = struct('file', {}, 'distance', {}, 'z', {}, 'genuine', {});
for q = 1:numel(queries)
    f = sig_features(queries{q});
    d = mean(arrayfun(@(i) cos_dist(f, R(i, :)), 1:n));
    z = (d - mu) / sigma;
    [~, name, ext] = fileparts(queries{q});
    results(end+1) = struct('file', [name ext], 'distance', d, 'z', z, 'genuine', z <= k); %#ok<AGROW>
    if z <= k, verdict = 'GENUINE'; else, verdict = 'FORGERY / different writer'; end
    fprintf('  %-24s distance=%.4f z=%+6.2f -> %s\n', [name ext], d, z, verdict);
end
end

function f = sig_features(path)
img = imread(path);
if size(img, 3) == 3, img = rgb2gray(img); end
img = imgaussfilt(img, 0.8);
bw = ~imbinarize(img);                       % ink = 1
[r, c] = find(bw);
bw = bw(min(r):max(r), min(c):max(c));
aspect = size(bw, 2) / size(bw, 1);
scale = min(256 / size(bw, 2), 128 / size(bw, 1));
bw = imresize(bw, scale) > 0.5;
canvas = false(128, 256);
oy = floor((128 - size(bw, 1)) / 2); ox = floor((256 - size(bw, 2)) / 2);
canvas(oy+1:oy+size(bw, 1), ox+1:ox+size(bw, 2)) = bw;

fh = extractHOGFeatures(canvas, 'CellSize', [16 16]);
fl = extractLBPFeatures(canvas, 'CellSize', [32 32], 'Upright', true);
[yy, xx] = find(canvas);
fg = [aspect / 4, mean(canvas(:)) * 5, mean(xx) / 256, mean(yy) / 128];
f = [fh / (norm(fh) + eps), fl / (norm(fl) + eps), fg];
end

function d = cos_dist(a, b)
d = 1 - (a * b') / (norm(a) * norm(b) + eps);
end

function files = list_images(folder)
s = [dir(fullfile(folder, '*.png')); dir(fullfile(folder, '*.jpg'))];
files = sort(fullfile({s.folder}, {s.name}));
files = files(:);
end
