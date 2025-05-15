    close all
    clear all
    
    % Ruta principal de la carpeta que contiene las subcarpetas
    mainFolder = 'F:/Documents/vc/cartoon/TRAIN';
    
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
    targetSize = [128, 128]; % [altura, anchura]
    
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
    
    %% Calcular histogramas bidimensionales de r y g normalizados
    fprintf('Calculando histogramas 2D de r-g normalizados para imágenes de entrenamiento...\n');
    numBins = 14; % Número de bins para el histograma 2D
    trainHistograms2D = cell(trainSize, 1);
    for i = 1:trainSize
        trainHistograms2D{i} = calcularHistogramasRG(trainImagesResized{i}, numBins);
    end
    
    fprintf('Calculando histogramas 2D de r-g normalizados para imágenes de prueba...\n');
    testHistograms2D = cell(testSize, 1);
    for i = 1:testSize
        testHistograms2D{i} = calcularHistogramasRG(testImagesResized{i}, numBins);
    end
    
    % Convertir los histogramas 2D a matrices de características
    totalBins = numBins * numBins; % Número total de bins en el histograma 2D
    trainFeatures = zeros(trainSize, totalBins);
    for i = 1:trainSize
        % Convertir la matriz 2D en un vector unidimensional
        trainFeatures(i, :) = trainHistograms2D{i}(:)';
    end
    
    testFeatures = zeros(testSize, totalBins);
    for i = 1:testSize
        testFeatures(i, :) = testHistograms2D{i}(:)';
    end
    
    fprintf('Descriptores de histogramas 2D calculados con éxito.\n');
    fprintf('  - Dimensión del histograma 2D: %dx%d bins\n', numBins, numBins);
    fprintf('  - Dimensión del vector característico: %d\n', totalBins);
    
    % Actualizar visualización para histograma 2D
    numVisualizar = min(3, trainSize);
    figure('Name', 'Histogramas 2D de r-g normalizados', 'Position', [100, 100, 800, 300*numVisualizar]);
    
    for i = 1:numVisualizar
        % Mostrar la imagen original
        subplot(numVisualizar, 2, (i-1)*2 + 1);
        imshow(trainImagesResized{i});
        title(sprintf('Imagen de %s', trainLabels{i}));
        
        % Mostrar el histograma 2D
        subplot(numVisualizar, 2, (i-1)*2 + 2);
        imagesc(trainHistograms2D{i});
        colormap(jet);
        colorbar;
        title('Histograma 2D (r-g normalizados)');
        xlabel('r normalizado');
        ylabel('g normalizado');
        axis square;
    end
    
    %% Mostrar histogramas 2D promedio para cada categoría
    figure('Name', 'Histogramas 2D promedio por categoría', 'Position', [200, 200, 1000, 600]);
    
    % Número de categorías
    numCategories = length(subfolders);
    numCols = min(3, numCategories);
    numRows = ceil(numCategories / numCols);
    
    % Para cada categoría
    for catIdx = 1:numCategories
        % Encontrar todas las imágenes de esta categoría
        categoryImages = find(trainClassIDs == catIdx);
        
        % Calcular histograma 2D promedio
        avgHist2D = zeros(numBins, numBins);
        
        for j = 1:length(categoryImages)
            imgIdx = categoryImages(j);
            avgHist2D = avgHist2D + trainHistograms2D{imgIdx};
        end
        avgHist2D = avgHist2D / length(categoryImages);
        
        % Visualizar
        subplot(numRows, numCols, catIdx);
        imagesc(avgHist2D);
        colormap(jet);
        colorbar;
        title(sprintf('Promedio: %s', subfolders(catIdx).name));
        xlabel('r normalizado');
        ylabel('g normalizado');
        axis square;
    end
    
    %% Crear una nueva variable para la app que incluya las características y la categoría
    trainFeaturesConCategoria = [trainFeatures, trainClassIDs'];
    
    % También podemos crear una versión para los datos de prueba
    testFeaturesConCategoria = [testFeatures, testClassIDs'];
    
    % Mostrar información sobre la nueva matriz
    fprintf('Matriz de características con categoría creada:\n');
    fprintf('  - Tamaño: %d filas x %d columnas\n', size(trainFeaturesConCategoria, 1), size(trainFeaturesConCategoria, 2));
    fprintf('  - Columnas 1-%d: Histograma 2D (r-g) aplanado\n', totalBins);
    fprintf('  - Columna %d: ID de la categoría\n', totalBins+1);
    
    % Visualizar las primeras filas para verificar la estructura
    fprintf('\nPrimeras 5 filas de la matriz (truncadas):\n');
    disp(trainFeaturesConCategoria(1:min(5,size(trainFeaturesConCategoria,1)), [1:5, end-4:end]));
    fprintf('... (columnas omitidas para mayor claridad)\n');
    
    % Guardar la matriz para usarla en la app (opcional)
    save('datos_para_app.mat', 'trainFeaturesConCategoria', 'testFeaturesConCategoria', 'folderMapping');
    fprintf('\nDatos guardados en "datos_para_app.mat" para usar en la app\n');
    
    %% Crear testHistogramsRGconCategoria: histogramas 2D originales con categoría
    fprintf('Creando estructura de histogramas 2D con categorías...\n');
    
    % Crear una estructura para almacenar los histogramas 2D y sus categorías
    testHistogramsRGconCategoria = struct('histograma', {}, 'categoria', {});
    
    % Llenar la estructura con los histogramas de prueba y sus categorías
    for i = 1:testSize
        testHistogramsRGconCategoria(i).histograma = testHistograms2D{i};
        testHistogramsRGconCategoria(i).categoria = testClassIDs(i);
    end
    
    fprintf('Variable testHistogramsRGconCategoria creada con éxito.\n');
    fprintf('  - Contiene %d elementos\n', length(testHistogramsRGconCategoria));
    fprintf('  - Cada elemento tiene un histograma 2D %dx%d y su categoría asociada\n', numBins, numBins);
    
    % Mostrar algunos ejemplos para verificar
    fprintf('\nEjemplos de testHistogramsRGconCategoria:\n');
    for i = 1:min(3, testSize)
        fprintf('  - Elemento %d: Histograma de dimensión %dx%d, Categoría: %d (%s)\n', ...
            i, size(testHistogramsRGconCategoria(i).histograma, 1), ...
            size(testHistogramsRGconCategoria(i).histograma, 2), ...
            testHistogramsRGconCategoria(i).categoria, ...
            folderMapping.FolderName{testHistogramsRGconCategoria(i).categoria});
    end
    
    % Guardar la matriz para usarla en la app (opcional)
    save('datos_para_app.mat', 'trainFeaturesConCategoria', 'testFeaturesConCategoria', ...
        'testHistogramsRGconCategoria', 'folderMapping');
    fprintf('\nDatos guardados en "datos_para_app.mat" para usar en la app\n');
    
   %% Entrenamiento del modelo usando trainClassifier.m
fprintf('\n== Entrenamiento del Modelo ==\n');

% Preparar los datos para el entrenador de modelos
fprintf('Preparando datos de entrenamiento...\n');
% Los datos de entrenamiento deben estar en formato [características, categoría]
% trainFeaturesConCategoria ya está en este formato

% Llamar a la función trainClassifier para entrenar el modelo
fprintf('Entrenando el modelo de clasificación...\n');
[trainedClassifier, validationAccuracy] = trainClassifier(trainFeaturesConCategoria);

% Mostrar la precisión de validación obtenida durante el entrenamiento
fprintf('Modelo entrenado con éxito.\n');
fprintf('Precisión de validación durante el entrenamiento: %.2f%%\n', validationAccuracy * 100);

%% Evaluación del modelo entrenado con los datos de prueba
fprintf('\n== Evaluación del Modelo con Datos de Prueba ==\n');

% Extraer características de testHistogramsRGconCategoria
fprintf('Preparando datos de prueba para clasificación...\n');

% Obtener número de elementos y tamaño del histograma
numTestSamples = length(testHistogramsRGconCategoria);
histSize = size(testHistogramsRGconCategoria(1).histograma);
numFeatures = prod(histSize);

% Crear matriz de características y vector de etiquetas reales
testFeatureMatrix = zeros(numTestSamples, numFeatures);
testLabelsActual = zeros(numTestSamples, 1);

% Llenar las matrices con los datos
for i = 1:numTestSamples
    % Aplanar el histograma 2D y guardarlo como vector de características
    testFeatureMatrix(i, :) = testHistogramsRGconCategoria(i).histograma(:)';
    
    % Guardar la categoría real
    testLabelsActual(i) = testHistogramsRGconCategoria(i).categoria;
end

% Realizar predicciones usando el modelo entrenado
fprintf('Realizando predicciones con el modelo entrenado...\n');

% Realizar predicciones según el tipo de modelo que devuelve trainClassifier
if isstruct(trainedClassifier) && isfield(trainedClassifier, 'predictFcn')
    % Si es un modelo exportado de Classification Learner
    testLabelsPredicted = trainedClassifier.predictFcn(testFeatureMatrix);
    fprintf('Predicción realizada usando trainedClassifier.predictFcn().\n');
else
    % Si es un objeto de modelo estándar
    testLabelsPredicted = predict(trainedClassifier, testFeatureMatrix);
    fprintf('Predicción realizada usando la función predict() estándar.\n');
end

% Calcular matriz de confusión
numCategories = length(unique(testLabelsActual));
confusionMatrix = zeros(numCategories);

for i = 1:numTestSamples
    actualID = testLabelsActual(i);
    predictedID = testLabelsPredicted(i);
    confusionMatrix(actualID, predictedID) = confusionMatrix(actualID, predictedID) + 1;
end

% Calcular métricas de rendimiento
correctPredictions = sum(testLabelsPredicted == testLabelsActual);
accuracy = correctPredictions / numTestSamples * 100;

% Mostrar resultados
fprintf('\nResultados de la clasificación:\n');
fprintf('  - Precisión de validación en entrenamiento: %.2f%%\n', validationAccuracy * 100);
fprintf('  - Total de muestras de prueba: %d\n', numTestSamples);
fprintf('  - Predicciones correctas: %d\n', correctPredictions);
fprintf('  - Precisión global en test: %.2f%%\n', accuracy);

% Mostrar matriz de confusión
fprintf('\nMatriz de Confusión:\n');
disp(confusionMatrix);

% Mostrar matriz de confusión visual
figure('Name', 'Matriz de Confusión', 'Position', [100, 100, 800, 600]);

% Crear etiquetas más legibles con ID y nombre de categoría
categoryLabels = cell(numCategories, 1);
for i = 1:numCategories
    if i <= height(folderMapping)
        categoryLabels{i} = sprintf('%d: %s', i, folderMapping.FolderName{i});
    else
        categoryLabels{i} = sprintf('%d', i);
    end
end

% Visualizar matriz de confusión normalizada por filas
confMatNorm = confusionMatrix ./ sum(confusionMatrix, 2);

% Detectar NaN debido a división por cero y reemplazarlos por ceros
confMatNorm(isnan(confMatNorm)) = 0;

heatmap(categoryLabels, categoryLabels, confMatNorm, ...
    'Colormap', jet, 'ColorbarVisible', 'on', ...
    'XLabel', 'Categoría Predicha', 'YLabel', 'Categoría Real');
title(sprintf('Matriz de Confusión (Norm.) - Precisión: %.1f%%', accuracy));

% Mostrar ejemplos de clasificación correcta e incorrecta
fprintf('\nBuscando ejemplos de clasificación correcta e incorrecta...\n');

% Encontrar índices de predicciones correctas e incorrectas
correctIndices = find(testLabelsPredicted == testLabelsActual);
incorrectIndices = find(testLabelsPredicted ~= testLabelsActual);

% Mostrar histogramas ejemplos (tanto correctos como incorrectos)
numExamples = min(3, min(length(correctIndices), length(incorrectIndices)));

if numExamples > 0
    figure('Name', 'Ejemplos de Clasificación', 'Position', [250, 250, 1200, 400*numExamples]);
    
    for i = 1:numExamples
        % Ejemplo de clasificación correcta
        if ~isempty(correctIndices)
            correctIndex = correctIndices(i);
            subplot(numExamples, 2, (i-1)*2+1);
            imagesc(testHistogramsRGconCategoria(correctIndex).histograma);
            colormap(jet); colorbar;
            title(sprintf('CORRECTO - Real: %s, Pred: %s', ...
                folderMapping.FolderName{testLabelsActual(correctIndex)}, ...
                folderMapping.FolderName{testLabelsPredicted(correctIndex)}));
            axis square;
        end
        
        % Ejemplo de clasificación incorrecta
        if ~isempty(incorrectIndices) && i <= length(incorrectIndices)
            incorrectIndex = incorrectIndices(i);
            subplot(numExamples, 2, (i-1)*2+2);
            imagesc(testHistogramsRGconCategoria(incorrectIndex).histograma);
            colormap(jet); colorbar;
            title(sprintf('INCORRECTO - Real: %s, Pred: %s', ...
                folderMapping.FolderName{testLabelsActual(incorrectIndex)}, ...
                folderMapping.FolderName{testLabelsPredicted(incorrectIndex)}));
            axis square;
        end
    end
else
    fprintf('No hay suficientes ejemplos para visualizar.\n');
end

% Guardar el modelo entrenado para uso futuro
save('modelo_entrenado.mat', 'trainedClassifier', 'validationAccuracy', 'numBins');
fprintf('\nModelo guardado en "modelo_entrenado.mat" para uso futuro.\n');