function [data, color] = load_data(name)
	filename = ['elliptical', 'ds577', '2d-4c-no0', '2d-4c-no1', '2d-4c-no4', ...
		    'adult', 'bank', 'census1990', 'creditcard', 'diabetic'];
	name = lower(strtrim(name));
	if ~any(contains(filename, name))
		fprintf('Not find this dataset');
		return;
	end
	data = load('../datasets/' + name + '.txt');
	color = load('../datasets/' + name + '_color.txt');
end
