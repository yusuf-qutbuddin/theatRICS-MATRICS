function [sdG,tau]=multiple_tau_SD_ACF(tr)

%make sure tr is a column vector
if size(tr,1)==1 && size(tr,2)>1
    tr=tr';
end




%% some definitions
tr_fluctutation = tr - mean(tr);
step=2^4;
binning=0;


numtr=numel(tr);                            %number of points in the trace
n_segments = 10;
segment_length = floor(numtr ./ n_segments);        % we split trace into 10 segments for SD calculation
segment_bounds = (0:(n_segments-1)) .* segment_length + 1;
numpts=floor((log2(segment_length/step)+1))*step+1;  %estimated number of points in autocorrelation function

%% allocate some space
segment_G = zeros(numpts,10);      %autocorrelation function 
tau = zeros(numpts,1);    %lag time in units of bins

for i_segment = 1:n_segments % Loop over segments
    cnt = 0;                  %counter
    lower_segment_bound = segment_bounds(i_segment);
    upper_segment_bound = min([lower_segment_bound + segment_length, numtr]);
    tr_segment = tr(lower_segment_bound:upper_segment_bound);
    tr_fluc_segment = tr_fluctutation(lower_segment_bound:upper_segment_bound);
    segment_length_copy = segment_length; % We need a new working copy for each segment, as this thing is overwritten a few times
    %% calculate the first 8 points of the autocorrelation function
    for i=1:step % Loop over edges within cascade 1
        shiftwidth=i-1;             %that's kind of the lag time in bin increments
        if size(tr_fluc_segment,1)>shiftwidth
            cnt=cnt+1;
            segment_G(cnt, i_segment) = mean(tr_fluc_segment(1:(segment_length_copy-shiftwidth)).*tr_fluc_segment((1+shiftwidth):segment_length_copy)) ./...
                (mean(tr_segment(1:(segment_length_copy-shiftwidth))) .* mean(tr_segment((1+shiftwidth):segment_length_copy)));
            if i_segment == 1 % We need that only once
                tau(cnt)=(i-1)*2^binning;
            end % if i_segment == 1
        end % if size(tr,1)>shiftwidth
    end % for i=1:step

    %% stepwise increase of binwidth in powers of 2 (multiple tau)
    while segment_length_copy>step % Loop over cascades >1
        tau0=2^binning*step;
        taustep=2^binning;
        for i=1:step % Loop over edges in cascade
            shiftwidth=step+i-1;
            if size(tr_fluc_segment,1)>shiftwidth
                cnt=cnt+1;
                segment_G(cnt, i_segment) = mean(tr_fluc_segment(1:(segment_length_copy-shiftwidth)).*tr_fluc_segment((1+shiftwidth):segment_length_copy)) ./...
                    (mean(tr_segment(1:(segment_length_copy-shiftwidth))) .* mean(tr_segment((1+shiftwidth):segment_length_copy)));
                if i_segment == 1 % We need that only once
                    tau(cnt)=tau0+(i-1)*taustep;
                end % if i_segment == 1
            end % if size(tr_fluc_segment,1)>shiftwidth
        end % for i=1:step
        segment_length_copy=2*floor(segment_length_copy/2);  
        tr_fluc_segment=(tr_fluc_segment(1:2:segment_length_copy)+tr_fluc_segment(2:2:segment_length_copy));
        tr_segment=(tr_segment(1:2:segment_length_copy)+tr_segment(2:2:segment_length_copy));
        segment_length_copy=numel(tr_segment);
        binning=binning+1;
    end % while numtr>step
end % for i_segment = 1:n_segments

mean_segment_G = mean(segment_G, 2);
sdG = sqrt(sum((segment_G - mean_segment_G).^2, 2)) ./ (n_segments-1);


