function [trace_bleachcorr,fitparam]=get_bleachcorrection(trace_x, trace_y, bleachcorr_degree)

% Center in x and standardize in y to better condition fit
trace_x_cen = trace_x - mean(trace_x);
mean_trace_y = mean(trace_y);
std_trace_y = std(trace_y);
Z_trace_y = (trace_y - mean_trace_y) ./ std_trace_y;


% Fit trace with polynomial of user-defined degree
fitparam = polyfit(trace_x_cen, Z_trace_y, bleachcorr_degree);
Z_trace_y_fit = polyval(fitparam, trace_x_cen);

% Scale Z fit back to original data scale
trace_y_fit = Z_trace_y_fit .* std_trace_y + mean_trace_y;

% Perform correction
trace_bleachcorr = trace_y./sqrt(trace_y_fit./trace_y_fit(1)) + trace_y_fit(1).*(1 - sqrt(trace_y_fit./trace_y_fit(1)));


end


