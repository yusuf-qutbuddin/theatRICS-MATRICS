function [G,tau]=multiple_tau_autocorrelation_noisesupression(tr)

%make sure tr is a column vector
if size(tr,1)==1 && size(tr,2)>1
    tr=tr';
end

%% some definitions
meanval=mean(tr);
tr_fluctutation=tr-meanval;
step=2^4;
% binwidth=1;
binning=0;


numtr=numel(tr);                            %number of points in the trace
numpts=floor((log2(numtr/step)+1))*step+1;  %estimated number of points in autocorrelation function

%% allocate some space
G = zeros(numpts,1);      %autocorrelation function 
tau = zeros(numpts,1);    %lag time in units of bins
cnt = 0;                  %counter

%% calculate the first 8 points of the autocorrelation function
for i=1:step
    shiftwidth=i-1;             %that's kind of the lag time in bin increments
    if size(tr,1)>shiftwidth
        cnt=cnt+1;
        G(cnt) = mean(tr_fluctutation(1:(numtr-shiftwidth)).*tr_fluctutation((1+shiftwidth):numtr)) ./...
            (mean(tr(1:(numtr-shiftwidth))) .* mean(tr((1+shiftwidth):numtr)));
        tau(cnt)=(i-1)*2^binning;
    end
end

%% stepwise increase of binwidth in powers of 2 (multiple tau)
while numtr>step
    tau0=2^binning*step;
    taustep=2^binning;
    for i=1:step
        shiftwidth=step+i-1;
        if size(tr,1)>shiftwidth
            cnt=cnt+1;
        G(cnt) = mean(tr_fluctutation(1:(numtr-shiftwidth)).*tr_fluctutation((1+shiftwidth):numtr)) ./...
            (mean(tr(1:(numtr-shiftwidth))) .* mean(tr((1+shiftwidth):numtr)));
            tau(cnt)=tau0+(i-1)*taustep;
        end
    end
    numtr=2*floor(numtr/2);  
    tr_fluctutation=(tr_fluctutation(1:2:numtr)+tr_fluctutation(2:2:numtr));
    tr=(tr(1:2:numtr)+tr(2:2:numtr));
    numtr=numel(tr);
%     binwidth=2*binwidth;
    binning=binning+1;
end
