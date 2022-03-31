function [out] = normal(edges)
% edges is a double images
% edges = 255 * im2double(edges)
edges = 255 * im2double(edges);
[n, m] = size(edges);
out = zeros(m, n);
mini = min(min(edges));
maxi = max(max(edges));
for i=1:n
    for j=1:m
        out(i, j) = floor(((edges(i, j) - mini)*255)/(maxi - mini));
    end
end
% out =  * out;
out = uint8(out);

end

