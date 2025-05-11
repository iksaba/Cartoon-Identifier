close all

% Ruta principal de la carpeta que contiene las subcarpetas
mainFolder = '/MATLAB Drive/Cartoon-Identifier/TRAIN';

% Obtener todas las subcarpetas
subfolders = dir(mainFolder);
subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

% Crear un mapeo de nombres de carpetas a identificadores numéricos
folderMapping = table();
folderMapping.FolderName = {subfolders.name}';
folderMapping.FolderID = (1:length(subfolders))';

% Mostrar la tabla de mapeo
disp('Mapeo de Carpetas a Identificadores:');
disp(folderMapping);

% Inicializar las listas para las imágenes
imageFiles = {};
imageLabels = {};
imageClassIDs = [];

% Definir el tamaño estándar para todas las imágenes
targetSize = [360, 640]; % [altura, anchura]

% Recorrer todas las subcarpetas y obtener las imágenes
for k = 1:length(subfolders)
    folderPath = fullfile(mainFolder, subfolders(k).name);
    images = dir(fullfile(folderPath, '*.jpg'));
    
    % Agregar las imágenes a la lista
    for i = 1:length(images)
        imageFiles{end+1} = fullfile(images(i).folder, images(i).name);
        imageLabels{end+1} = subfolders(k).name;
        imageClassIDs(end+1) = k; % Usar el índice como identificador numérico
    end
end

% Número total de imágenes
numImages = length(imageFiles);

% Aleatorizar las imágenes (manteniendo la correspondencia entre imágenes, etiquetas e IDs)
rng('default'); % Para reproducibilidad
indices = randperm(numImages);
imageFiles = imageFiles(indices);
imageLabels = imageLabels(indices);
imageClassIDs = imageClassIDs(indices);

% Definir los tamaños para el conjunto de entrenamiento y prueba
trainSize = round(0.7 * numImages);  % 70% para entrenamiento
testSize = numImages - trainSize;    % 30% para prueba

% Dividir las imágenes en conjuntos de entrenamiento y prueba
trainImages = imageFiles(1:trainSize);
trainLabels = imageLabels(1:trainSize);
trainClassIDs = imageClassIDs(1:trainSize);
testImages = imageFiles(trainSize+1:end);
testLabels = imageLabels(trainSize+1:end);
testClassIDs = imageClassIDs(trainSize+1:end);

% Mostrar información de la división
fprintf('Total de imágenes: %d\n', numImages);
fprintf('Imágenes para entrenamiento: %d (70%%)\n', trainSize);
fprintf('Imágenes para prueba: %d (30%%)\n', testSize);
fprintf('Tamaño estándar de imágenes: %dx%d píxeles\n', targetSize(2), targetSize(1));

% Información de categorías
fprintf('Categorías detectadas: %d\n', length(subfolders));
for i = 1:length(subfolders)
    fprintf('  - ID %d: %s (%d imágenes)\n', i, subfolders(i).name, ...
        sum(strcmp(imageLabels, subfolders(i).name)));
end

%% Cargar y redimensionar todas las imágenes
fprintf('Cargando y redimensionando imágenes de entrenamiento...\n');
trainImagesResized = cell(trainSize, 1);
for i = 1:trainSize
    originalImg = imread(trainImages{i});
    trainImagesResized{i} = imresize(originalImg, targetSize);
    if mod(i, 50) == 0
        fprintf('  Procesadas %d/%d imágenes de entrenamiento\n', i, trainSize);
    end
end

fprintf('Cargando y redimensionando imágenes de prueba...\n');
testImagesResized = cell(testSize, 1);
for i = 1:testSize
    originalImg = imread(testImages{i});
    testImagesResized{i} = imresize(originalImg, targetSize);
    if mod(i, 50) == 0
        fprintf('  Procesadas %d/%d imágenes de prueba\n', i, testSize);
    end
end

%% Calcular histogramas separados para los canales R y B
fprintf('Calculando histogramas para canales R y B de imágenes de entrenamiento...\n');
numBins = 64; % Número de bins para cada histograma
[trainHistogramsR, trainHistogramsB, trainHistogramsG] = calcularHistogramasRBG(trainImagesResized, numBins);

fprintf('Calculando histogramas para canales R y B de imágenes de prueba...\n');
[testHistogramsR, testHistogramsB, testHistogramsG] = calcularHistogramasRBG(testImagesResized, numBins);

% Convertir los histogramas a matrices de características
trainFeatures = zeros(trainSize, numBins * 3); % 2 histogramas (R y B) de numBins cada uno
for i = 1:trainSize
    % Concatenar los dos histogramas (R y B) en un vector
    trainFeatures(i, :) = [trainHistogramsR{i}, trainHistogramsB{i}, trainHistogramsG{i}];
end

testFeatures = zeros(testSize, numBins *3);
for i = 1:testSize
    testFeatures(i, :) = [testHistogramsR{i}, testHistogramsB{i}, trainHistogramsG{i}];
end

fprintf('Descriptores de histogramas R y B calculados con éxito.\n');
fprintf('  - Dimensión de cada histograma: %d bins\n', numBins);
fprintf('  - Dimensión del vector característico: %d\n', numBins * 2);

%% VISUALIZACIÓN DE EJEMPLO
% Visualizar los histogramas R y B para algunas imágenes
numVisualizar = min(3, trainSize);
figure('Name', 'Histogramas de Canales R y B', 'Position', [100, 100, 1000, 300*numVisualizar]);

for i = 1:numVisualizar
    % Mostrar la imagen original
    subplot(numVisualizar, 4, (i-1)*4 + 1);
    imshow(trainImagesResized{i});
    title(sprintf('Imagen de %s', trainLabels{i}));
    
    % Mostrar el histograma del canal R
    subplot(numVisualizar, 4, (i-1)*4 + 2);
    bar(trainHistogramsR{i}, 'r');
    title('Histograma Canal R (normalizado)');
    xlabel('Bin (proporción de rojo)');
    ylabel('Frecuencia');
    xlim([1 numBins]);
    
    % Mostrar el histograma del canal B
    subplot(numVisualizar, 4, (i-1)*4 + 3);
    bar(trainHistogramsB{i}, 'b');
    title('Histograma Canal B (normalizado)');
    xlabel('Bin (proporción de azul)');
    ylabel('Frecuencia');
    xlim([1 numBins]);

    % Mostrar el histograma del canal G
    subplot(numVisualizar, 4, (i-1)*4 + 4);
    bar(trainHistogramsG{i}, 'g');
    title('Histograma Canal G (normalizado)');
    xlabel('Bin (proporción de verde)');
    ylabel('Frecuencia');
    xlim([1 numBins]);
end

%% Mostrar histogramas promedio para cada categoría
figure('Name', 'Histogramas promedio por categoría', 'Position', [200, 200, 1000, 600]);

% Número de categorías
numCategories = length(subfolders);
numCols = min(3, numCategories);
numRows = ceil(numCategories / numCols);

% Para cada categoría
for catIdx = 1:numCategories
    % Encontrar todas las imágenes de esta categoría
    categoryImages = find(trainClassIDs == catIdx);
    
    % Calcular histogramas promedio (R y B)
    avgHistR = zeros(1, numBins);
    avgHistB = zeros(1, numBins);
    avgHistG = zeros(1, numBins);
    
    for j = 1:length(categoryImages)
        imgIdx = categoryImages(j);
        avgHistR = avgHistR + trainHistogramsR{imgIdx};
        avgHistB = avgHistB + trainHistogramsB{imgIdx};
        avgHistG = avgHistG + trainHistogramsG{imgIdx};
    end
    avgHistR = avgHistR / length(categoryImages);
    avgHistB = avgHistB / length(categoryImages);
    avgHistG = avgHistG / length(categoryImages);
    
    % Visualizar
    subplot(numRows, numCols, catIdx);
    hold on;
    bar(avgHistR, 'r', 'FaceAlpha', 0.5);
    bar(avgHistB, 'b', 'FaceAlpha', 0.5);
    bar(avgHistG, 'g', 'FaceAlpha', 0.5);
    hold off;
    title(sprintf('Promedio: %s', subfolders(catIdx).name));
    xlabel('Bin');
    ylabel('Frecuencia normalizada');
    legend('Canal R', 'Canal B', 'Canal G', 'Location', 'northeast');
    xlim([1 numBins]);
end

%% Crear una nueva variable para la app que incluya las características y la categoría
trainFeaturesConCategoria = [trainFeatures, trainClassIDs'];

% También podemos crear una versión para los datos de prueba
testFeaturesConCategoria = [testFeatures, testClassIDs'];

% Mostrar información sobre la nueva matriz
fprintf('Matriz de características con categoría creada:\n');
fprintf('  - Tamaño: %d filas x %d columnas\n', size(trainFeaturesConCategoria, 1), size(trainFeaturesConCategoria, 2));
fprintf('  - Columnas 1-%d: Histograma R normalizado\n', numBins);
fprintf('  - Columnas %d-%d: Histograma B normalizado\n', numBins+1, numBins*2);
fprintf('  - Columna %d: ID de la categoría\n', numBins*2+1);

% Visualizar las primeras filas para verificar la estructura
fprintf('\nPrimeras 5 filas de la matriz (truncadas):\n');
disp(trainFeaturesConCategoria(1:min(5,size(trainFeaturesConCategoria,1)), [1:5, numBins+1:numBins+5, end]));
fprintf('... (columnas omitidas para mayor claridad)\n');

% Guardar la matriz para usarla en la app (opcional)
save('datos_para_app.mat', 'trainFeaturesConCategoria', 'testFeaturesConCategoria', 'folderMapping');
fprintf('\nDatos guardados en "datos_para_app.mat" para usar en la app\n');

%% AQUÍ PUEDES AÑADIR TU CÓDIGO PARA ENTRENAR UN CLASIFICADOR
% Por ejemplo:
%
% % Entrenar un clasificador KNN
% fprintf('Entrenando clasificador KNN...\n');
% mdl = fitcknn(trainFeatures, trainClassIDs, 'NumNeighbors', 5);
%
% % Realizar predicciones
% predictedIDs = predict(mdl, testFeatures);
%
% % Calcular matriz de confusión y precisión
% confusionMatrix = zeros(length(subfolders));
% for i = 1:length(testClassIDs)
%     actualID = testClassIDs(i);
%     predictedID = predictedIDs(i);
%     confusionMatrix(actualID, predictedID) = confusionMatrix(actualID, predictedID) + 1;
% end
%
% % Visualizar resultados
% disp('Matriz de confusión:');
% disp(confusionMatrix);
%
% precisión = sum(diag(confusionMatrix)) / sum(confusionMatrix(:)) * 100;
% fprintf('Precisión global: %.2f%%\n', precisión);