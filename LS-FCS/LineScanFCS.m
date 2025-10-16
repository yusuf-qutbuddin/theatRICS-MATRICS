folder = {
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\Data\D044_LSM780_Feynman\20231221_JHK_GUVs_FCS_Coatings'
    };


file = {
    '20nmAu_GUV2_LS-FCS_ch1.tif'
    };


line_time = 472.73E-6; % Line time calibration in s

write_traces = true;
write_traces_skip = 10;

% Expand in case multiple files in one folder have been chosen
if length(folder) == 1 && length(file) > 1
    folder(2:length(file)) = folder(1);
end

for i_file = 1:numel(file)

    disp(['processing ' file{i_file} '...']);

    path = fullfile(folder{i_file}, file{i_file});
    tiff_obj = Tiff(path,'r');
    im = read(tiff_obj);
%     im = tiffreadVolume(path);
    sz = size(im);
    
    if length(sz) > 2
        % Multi-channel

        for i_channel = 1:sz(3)
    
            for i_px = 1:sz(2)
                disp(['pixel ' num2str(i_px) '...']);
    
                % Get time trace for this px, and correlate
                trace_y = double(im(:,i_px,i_channel));
                [G, ~] = multiple_tau_autocorrelation_noisesupression(trace_y);
                [sdG, tau] = multiple_tau_SD_ACF(trace_y);
                tau = tau .* line_time;
                G = G(1:length(tau));
                acr = mean(trace_y);
    
                % Export
                acr_col = [acr; acr; zeros(length(tau)-3, 1)]; 
                out_Kristine = [tau(2:end) G(2:length(tau)) acr_col sdG(2:end)];
                writematrix(out_Kristine, fullfile(folder{i_file}, strrep(file{i_file},'.tif', ['_ch' num2str(i_channel) '_px' num2str(i_px) '_ACF.csv'])));
    
                % Perform bleaching correction, and repeat
                [trace_y_blcorr, ~, bleachcorr_degree] = get_bleachcorrection_autodegree(1:sz(1), trace_y');
                disp(['bleachcorr_degree: ' num2str(bleachcorr_degree)]);
    
                [G_blcorr, ~] = multiple_tau_autocorrelation_noisesupression(trace_y_blcorr);
                [sdG_blcorr, tau] = multiple_tau_SD_ACF(trace_y_blcorr);
                tau = tau .* line_time;
                G_blcorr = G_blcorr(1:length(tau));
                acr_blcorr = mean(trace_y_blcorr);
    
                % Export
                acr_col_blcorr = [acr_blcorr; acr_blcorr; zeros(length(tau)-3, 1)]; 
                out_Kristine_blcorr = [tau(2:end) G_blcorr(2:length(tau)) acr_col_blcorr sdG_blcorr(2:end)];
                writematrix(out_Kristine_blcorr, fullfile(folder{i_file}, strrep(file{i_file},'.tif', ['_ch' num2str(i_channel) '_px' num2str(i_px) '_ACF_blcorr.csv'])));
                
    
            end % for i_px = 1:sz()
        end % for i_channel = 1:sz(3)

    else %  if length(sz) > 2
        % Single-channel
        for i_px = 1:sz(2)
            disp(['pixel ' num2str(i_px) '...']);

            % Get time trace for this px, and correlate
            trace_y = double(im(:,i_px));
            [G, ~] = multiple_tau_autocorrelation_noisesupression(trace_y);
            [sdG, tau] = multiple_tau_SD_ACF(trace_y);
            tau = tau .* line_time;
            G = G(1:length(tau));
            acr = mean(trace_y);

            % Export
            acr_col = [acr; acr; zeros(length(tau)-3, 1)]; 
            out_Kristine = [tau(2:end) G(2:length(tau)) acr_col sdG(2:end)];
            writematrix(out_Kristine, fullfile(folder{i_file}, strrep(file{i_file},'.tif', ['_px' num2str(i_px) '_ACF.csv'])));

            % Perform bleaching correction, and repeat
            [trace_y_blcorr, ~, bleachcorr_degree] = get_bleachcorrection_autodegree(1:sz(1), trace_y');
            disp(['bleachcorr_degree: ' num2str(bleachcorr_degree)]);

            [G_blcorr, ~] = multiple_tau_autocorrelation_noisesupression(trace_y_blcorr);
            [sdG_blcorr, tau] = multiple_tau_SD_ACF(trace_y_blcorr);
            tau = tau .* line_time;
            G_blcorr = G_blcorr(1:length(tau));
            acr_blcorr = mean(trace_y_blcorr);

            % Export
            acr_col_blcorr = [acr_blcorr; acr_blcorr; zeros(length(tau)-3, 1)]; 
            out_Kristine_blcorr = [tau(2:end) G_blcorr(2:length(tau)) acr_col_blcorr sdG_blcorr(2:end)];
            writematrix(out_Kristine_blcorr, fullfile(folder{i_file}, strrep(file{i_file},'.tif', ['_px' num2str(i_px) '_ACF_blcorr.csv'])));
       
            if write_traces && mod(i_px - 1, write_traces_skip) == 0
                writematrix([trace_y, trace_y_blcorr'], fullfile(folder{i_file}, strrep(file{i_file},'.tif', ['_px' num2str(i_px) '_traces.csv'])));
            end

        
        
        end % for i_px = 1:sz()

    end %  if length(sz) > 2

end % for i=1:numel(file)
