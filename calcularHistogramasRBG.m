function [histR, histB, histG] = calcularHistogramasRBG(imagenes, numBins)
% CALCULARHISTOGRAMASRB Calcula histogramas separados para canales R y B normalizados
%   [histR, histB] = CALCULARHISTOGRAMASRB(imagenes) calcula dos histogramas
%   separados: uno para la proporción normalizada de rojo y otro para la
%   proporción normalizada de azul.
%
%   [histR, histB] = CALCULARHISTOGRAMASRB(imagenes, numBins) especifica el número
%   de bins para cada histograma (por defecto 32).
%
%   Entradas:
%     imagenes - Una sola imagen o un array de celdas con múltiples imágenes RGB
%     numBins  - Número de bins para cada histograma (opcional, por defecto 32)
%
%   Salidas:
%     histR - Histograma del canal Rojo normalizado
%             Para una imagen: vector de tamaño [1, numBins]
%             Para múltiples imágenes: array de celdas con histogramas R
%     histB - Histograma del canal Azul normalizado
%             Para una imagen: vector de tamaño [1, numBins]
%             Para múltiples imágenes: array de celdas con histogramas B

    % Verificar argumentos de entrada
    if nargin < 1
        error('Es necesario proporcionar al menos una imagen como argumento.');
    end
    
    if nargin < 2
        numBins = 32; % Valor predeterminado: 32 bins
    end
    
    % Determinar si la entrada es una sola imagen o múltiples imágenes
    esCelda = iscell(imagenes);
    
    if esCelda
        % Procesar múltiples imágenes
        numImagenes = length(imagenes);
        histR = cell(numImagenes, 1);
        histB = cell(numImagenes, 1);
        histG = cell(numImagenes, 1);
        
        for i = 1:numImagenes
            [histR{i}, histB{i}, histG{i}] = calcularHistogramasRBSingleImg(imagenes{i}, numBins);
        end
    else
        % Procesar una sola imagen
        [histR, histB, histG] = calcularHistogramasRBSingleImg(imagenes, numBins);
    end
end

function [histR, histB, histG] = calcularHistogramasRBSingleImg(img, numBins)
    % Verificar que la imagen sea RGB
    if size(img, 3) ~= 3
        error('La imagen debe ser RGB (3 canales)');
    end
    
    % Convertir a double para cálculos precisos
    if ~isa(img, 'double')
        img = double(img);
        if max(img(:)) > 1
            img = img / 255; % Normalizar a [0,1] si es necesario
        end
    end
    
    % Obtener los canales RGB
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    
    % Calcular la suma de valores RGB para cada píxel
    suma = R + G + B;
    
    % Evitar división por cero
    suma(suma == 0) = 1;
    
    % Normalizar los canales (dividir cada canal por la suma RGB)
    R_norm = R ./ suma;
    B_norm = B ./ suma;
    G_norm = G ./ suma;
    
    % Convertir a vectores
    R_vec = R_norm(:);
    B_vec = B_norm(:);
    G_vec = G_norm(:);
    
    % Definir los bordes para los histogramas (entre 0 y 1 para valores normalizados)
    edges = linspace(0, 1, numBins+1);
    
    % Crear histogramas para cada canal
    histR = histcounts(R_vec, edges, 'Normalization', 'probability');
    histB = histcounts(B_vec, edges, 'Normalization', 'probability');
    histG = histcounts(G_vec, edges, 'Normalization', 'probability');
end