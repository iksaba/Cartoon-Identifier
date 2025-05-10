close all

% Ruta principal de la carpeta que contiene las subcarpetas
mainFolder = 'C:\Users\iker.santin\Downloads\TRAIN\TRAIN';

% Obtener todas las subcarpetas
subfolders = dir(mainFolder);
subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

% Inicializar las listas para las imágenes
imageFiles = {};

% Recorrer todas las subcarpetas y obtener las imágenes
for k = 1:length(subfolders)
    folderPath = fullfile(mainFolder, subfolders(k).name);
    images = dir(fullfile(folderPath, '*.jpg'));
    
    % Agregar las imágenes a la lista
    for i = 1:length(images)
        imageFiles{end+1} = fullfile(images(i).folder, images(i).name);
    end
end

% Número total de imágenes
numImages = length(imageFiles);

% Aleatorizar las imágenes
indices = randperm(numImages);

% Definir los tamaños para el conjunto de entrenamiento y prueba
trainSize = round(0.7 * numImages);  % 70% para entrenamiento
testSize = numImages - trainSize;    % 30% para prueba

% Dividir las imágenes en dos conjuntos
trainImages = imageFiles(indices(1:trainSize));
testImages = imageFiles(indices(trainSize+1:end));

% Número de descriptores que se van a calcular
N = 30;
descriptorsPerSeries = struct();

% Calcular los descriptores para cada serie
for k = 1:length(subfolders)
    % Reemplazar espacios con guiones bajos para hacer los nombres válidos
    validFieldName = strrep(subfolders(k).name, ' ', '_');  
    
    % Crear la estructura para almacenar los descriptores de esta serie
    descriptorsPerSeries.(validFieldName) = [];
    
    folderPath = fullfile(mainFolder, subfolders(k).name);
    images = dir(fullfile(folderPath, '*.jpg'));
    
    for i = 1:length(images)
        filename = fullfile(images(i).folder, images(i).name);
        I = rgb2gray(imread(filename));    % Convertir a escala de grises
        BW = I < 160;                      % Umbralizar la imagen
        BW = imfill(BW, 'holes');          % Rellenar agujeros
        BW = bwareafilt(BW, 1);            % Filtrar por el área más grande
        
        % Obtener los descriptores
        F = MyFourierDescriptors(BW, N);
        descriptorsPerSeries.(validFieldName) = [descriptorsPerSeries.(validFieldName); F]; % Almacenar
    end
end

% Función para calcular los descriptores de Fourier
function descriptors = MyFourierDescriptors(BW, N)
    % Calcula los descriptores de Fourier para una imagen binaria
    % BW: Imagen binaria
    % N: Número de descriptores a calcular
    
    % Calcular la transformada de Fourier 2D de la imagen binaria
    F = fft2(BW);
    
    % Obtener los coeficientes de Fourier
    F = fftshift(F); % Centrar la transformada
    descriptors = zeros(1, N);
    
    % Tomar los primeros N coeficientes
    for i = 1:N
        descriptors(i) = abs(F(i));
    end
end

% Inicializar la matriz de confusión
confusionMatrix = zeros(length(subfolders), length(subfolders));

% Repetir para todas las imágenes del conjunto de test
for i = 1:length(testImages)
    % Obtener la imagen de test y calcular sus descriptores
    testImagePath = testImages{i};
    testImage = imread(testImagePath);
    I_test = rgb2gray(testImage);
    BW_test = I_test < 160;
    BW_test = imfill(BW_test, 'holes');
    BW_test = bwareafilt(BW_test, 1);
    F_test = MyFourierDescriptors(BW_test, N);
    
    % Inicializar las distancias para esta imagen de prueba
    distances = zeros(1, length(subfolders));
    
    % Comparar las características del fotograma de prueba con las de las series
    for k = 1:length(subfolders)
        validFieldName = strrep(subfolders(k).name, ' ', '_');  % Reemplazar espacios con guiones bajos
        
        seriesDescriptors = descriptorsPerSeries.(validFieldName);  % Acceder usando el nombre válido
        
        % Calcular la distancia Euclidiana entre los descriptores de prueba y los de la serie
        for j = 1:size(seriesDescriptors, 1)
            distances(k) = distances(k) + sum((seriesDescriptors(j, :) - F_test).^2);
        end
    end
    
    % Encontrar la serie con la menor distancia
    [~, minIdx] = min(distances);
    identifiedSeries = subfolders(minIdx).name;
    
    % Obtener la etiqueta correcta (serie de la imagen)
    [~, imageFolderName] = fileparts(fileparts(testImagePath));  % Obtener el nombre de la carpeta de la serie
    actualSeriesIdx = find(strcmp({subfolders.name}, imageFolderName));  % Buscar el índice de la serie en subfolders
    actualSeries = subfolders(actualSeriesIdx).name;  % Obtener el nombre de la serie correctamente

    % Actualizar la matriz de confusión
    actualIdx = find(strcmp({subfolders.name}, actualSeries));
    confusionMatrix(actualIdx, minIdx) = confusionMatrix(actualIdx, minIdx) + 1;
end

% Mostrar la matriz de confusión
disp('Matriz de Confusión:');
disp(confusionMatrix);

