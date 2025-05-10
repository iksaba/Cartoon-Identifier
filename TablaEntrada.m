function [imageTable, classTable] = createImageDatasetTable(mainFolder)
    % Crea una tabla con información de las imágenes y otra con las clases
    % mainFolder: Ruta principal que contiene las subcarpetas por clase
    
    % Obtener todas las subcarpetas (clases)
    subfolders = dir(mainFolder);
    subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));
    
    % Crear tabla de clases
    classTable = table((1:length(subfolders))', {subfolders.name}', ...
                      'VariableNames', {'ClassID', 'ClassName'});
    
    % Inicializar lista para la tabla de imágenes
    imageData = cell(0, 4); % 4 columnas: FilePath, FileName, ClassID, IsTest
    
    % Recorrer todas las subcarpetas
    for classID = 1:length(subfolders)
        folderPath = fullfile(mainFolder, subfolders(classID).name);
        images = dir(fullfile(folderPath, '*.jpg'));
        
        % Aleatorizar las imágenes de esta clase
        imgIndices = randperm(length(images));
        
        % Determinar punto de división (70% aprendizaje, 30% test)
        splitPoint = round(0.7 * length(images));
        
        % Procesar imágenes de aprendizaje (0)
        for i = 1:splitPoint
            imgIdx = imgIndices(i);
            imageData(end+1, :) = {fullfile(images(imgIdx).folder, images(imgIdx).name), classID, 0}; % IsTest = 0 para aprendizaje
        end
        
        % Procesar imágenes de test (1)
        for i = splitPoint+1:length(images)
            imgIdx = imgIndices(i);
            imageData(end+1, :) = {fullfile(images(imgIdx).folder, images(imgIdx).name),classID, 1}; % IsTest = 1 para test
        end
    end
    
    % Convertir a tabla con nombres de columnas correctos
    imageTable = cell2table(imageData, 'VariableNames', {'FilePath', 'FileName', 'ClassID', 'IsTest'});
    
    % Mezclar todas las filas para mayor aleatoriedad
    imageTable = imageTable(randperm(height(imageTable)), :);
    
    % Mostrar resumen
    disp('Resumen del dataset creado:');
    disp(['Total imágenes: ', num2str(height(imageTable))]);
    disp(['Aprendizaje (0): ', num2str(sum(imageTable.IsTest == 0)), ...
         ' (', num2str(round(mean(imageTable.IsTest == 0)*100), '%)')]);
    disp(['Test (1): ', num2str(sum(imageTable.IsTest == 1)), ...
         ' (', num2str(round(mean(imageTable.IsTest == 1)*100), '%)')]);
    
    % Mostrar distribución por clase
    figure;
    histogram(imageTable.ClassID);
    title('Distribución de imágenes por clase');
    xlabel('Class ID');
    ylabel('Número de imágenes');
end

mainFolder = 'C:\Users\iker.santin\Downloads\TRAIN\TRAIN';
[imageTable, classTable] = createImageDatasetTable(mainFolder);