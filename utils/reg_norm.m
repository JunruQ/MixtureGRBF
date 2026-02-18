function mat_z_score = reg_norm(dataset_name,input)

switch dataset_name
    case 'Tadpole'
        if exist('./input/Tadpole/TADPOLE_input_pass_z.csv','file')
            reg_data = readtable('./input/Tadpole/TADPOLE_input_pass_z.csv','VariableNamingRule','preserve');
            mat_z_score = reg_data{:,11:end};
        else
            mat_z_score = reg_norm_tadpole(input);
        end
end
end



