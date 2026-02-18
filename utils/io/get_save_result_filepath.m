function [save_dir1, save_dir] = get_save_result_filepath(options, nsubtype)
    save_dir = ['./output/',options.output_file_name];
    save_dir1 = [save_dir, '/',num2str(nsubtype),'_subtypes'];
    if ~exist(save_dir1, 'dir')
        mkdir(save_dir1)
end
