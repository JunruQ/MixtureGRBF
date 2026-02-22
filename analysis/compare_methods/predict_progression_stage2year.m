input = readtable("./input/ADNI/ADNI_demo_MRI_PET_ro_11_19.csv","VariableNamingRule","preserve");
subtype_stage = readtable("./output/FTR_fsx/3_subtypes/subtype_stage.csv");
joindata = outerjoin(input,subtype_stage,'MergeKeys',true);

dataTable = joindata(:,["RID","diagnosis","years","stage","subtype"]);
dataTable = rmmissing(dataTable);

% Find unique 'rid' values where 'diagnosis' equals 1
RID_AD = unique(dataTable.RID(dataTable.diagnosis == 1));

% Filter rows where 'rid' is in the list of 'rids_with_diagnosis_1'
dataAD = dataTable(ismember(dataTable.RID, RID_AD), :);
dataAD = dataAD(dataAD.subtype==1,:);

% Sort rows by 'RID' column and then by 'years' column
sorted_dataAD = sortrows(dataAD, {'RID', 'years'});

% Generate stages from 0 to 1 in increments of 0.1
step = 0.1;
interpolated_stages = 0:step:1;

% Initialize a matrix to store interpolated values
interpolated_matrix = zeros(height(sorted_dataAD),length(interpolated_stages));
diff_matrix = zeros(height(sorted_dataAD));
diff_year = zeros(height(sorted_dataAD),1);


% Loop through each row in dataAD
for i = 1:height(sorted_dataAD)
    % Get the stage value for the i-th row
    stage_value = sorted_dataAD.stage(i);
    
    % Calculate indices a and b
    a = floor(stage_value / step);
    b = ceil(stage_value / step);
    
    % Calculate interpolation values based on the formula
    if a == b
        interpolated_matrix(i, a+1) = 1;
    else
        c = (b * 0.1 - stage_value) / (b * 0.1 - a * 0.1);
        interpolated_matrix(i, a+1) = c;
        interpolated_matrix(i, b+1) = 1 - c;
    end

    if i > 1 
        if  sorted_dataAD.RID(i) ==  sorted_dataAD.RID(i-1)
            diff_matrix(i,i) = 1;
            diff_matrix(i,i-1) = -1;
            diff_year(i) = sorted_dataAD.years(i) - sorted_dataAD.years(i-1);
        end
    end
end 

m = 1/step;

A = diff_matrix * interpolated_matrix(:,2:end);
b = diff_year;
D = zeros(m-1,m);
for i = 1:m-1
    D(i,i) = 1;
    D(i,i+1) = -1;
end
c = zeros(m-1,1);

s = quadprog(A'*A, -A'*b, D,c,[],[],zeros(m,1));
plot(0:step:1,[0;s])

