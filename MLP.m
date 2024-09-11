
%----PARAMS------%
non_linear_data = true;
if non_linear_data
    mA = [ 0.5, 0.3];    
    sigmaA = 0.2;
    mB = [ 0.0, -0.1];   
    sigmaB = 0.3;
else
    mA = [0.5 , 0.5]; 
    sigmaA = 0.3;
    mB = [0 , 0.0];
    sigmaB = 0.2;
end

n = 100;
nodes = 40;
alpha = 0.9;
eta = 0.05;
epochs = 200;

ratioA = 0.5;
ratioB = 1;

numTrainA = round(ratioA * n);
numTrainB = round(ratioB * n);
%----------------%

%----MAIN--------- %
[classA, classB] = generateClasses(n , mA, mB, sigmaA, sigmaB, non_linear_data,ratioA, ratioB );
n = (numTrainA + numTrainB) / 2;
%disp(n);
n_test = (100 - n);
%disp(n_test)
[classA_train, classA_test, classB_train, classB_test] = splitClassesUnequal(classA, classB, numTrainA, numTrainB);

[X,y,W1, W2, dW1, dW2] = generateData(classA_train, classB_train, n, nodes);
[X_test,y_test,~,~,~, ~] = generateData(classA_test, classB_test, n_test, nodes);

[W1Star, W2Star] = MLP(X,X_test, y, y_test, W1,W2,nodes, n, epochs, alpha, eta, dW1,dW2,n_test);
plotData(X, y, W1, W2, classA_train, classB_train)
evluateClassifier(X_test, y_test, W1Star, W2Star, classA_test, classB_test, n_test);
%----------------%
function [classA_train, classA_test, classB_train, classB_test] = splitClassesUnequal(classA, classB, numTrainA, numTrainB)

    classA_neg = classA(:, classA(1,:) < 0); % classA(1,:) < 0
    classA_pos = classA(:, classA(1,:) > 0); % classA(1,:) > 0

    numTrainA_neg = round(0.8 * numTrainA); 
    numTrainA_pos = numTrainA - numTrainA_neg; 

    indicesA_neg = randperm(size(classA_neg, 2)); 
    indicesA_pos = randperm(size(classA_pos, 2));

    trainIndicesA_neg = indicesA_neg(1:numTrainA_neg);
    trainIndicesA_pos = indicesA_pos(1:numTrainA_pos);

    classA_train = [classA_neg(:, trainIndicesA_neg), classA_pos(:, trainIndicesA_pos)];
    testIndicesA_neg = indicesA_neg(numTrainA_neg+1:end);
    testIndicesA_pos = indicesA_pos(numTrainA_pos+1:end);

    classA_test = [classA_neg(:, testIndicesA_neg), classA_pos(:, testIndicesA_pos)];

    indicesB = randperm(size(classB, 2));
    trainIndicesB = indicesB(1:numTrainB);
    testIndicesB = indicesB(numTrainB+1:end);

    classB_train = classB(:, trainIndicesB);
    classB_test = classB(:, testIndicesB);
end

function [classA_train, classA_test, classB_train, classB_test] = splitClasses(classA, classB, numTrainA, numTrainB)
    nA = size(classA, 2);
    nB = size(classB, 2);
    
    indicesA = randperm(nA); 
    indicesB = randperm(nB);  
    trainIndicesA = indicesA(1:numTrainA);         
    testIndicesA = indicesA(numTrainA+1:end);      
    
    trainIndicesB = indicesB(1:numTrainB);         
    testIndicesB = indicesB(numTrainB+1:end);      
    
    classA_train = classA(:, trainIndicesA);
    classA_test = classA(:, testIndicesA);
    
    classB_train = classB(:, trainIndicesB);
    classB_test = classB(:, testIndicesB);
end

function [accuracy] = evluateClassifier(X, y,  W1Star, W2Star, classA, classB, n)
    [~, ~, ~, out] = forwardPass(X, W1Star, W2Star, n);
    
    predictions = sign(out);
    
    accuracy = sum(predictions == y) / length(y);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);

    plotData(X, y, W1Star, W2Star,classA, classB , out);
    
end

function loss = computeCost(out, y)

       loss = (1/2) * sum((out - y).^2) ;

end

function [hin, hout, oin, out] = forwardPass(X, W1, W2 , n)
    hin = W1 * X + ones(1,2*n);
    hout = 2 ./ (1 + exp(-hin)) - 1;

    oin = W2 * hout + ones(1, n*2);
    out = 2 ./ (1+exp(-oin)) - 1;

end

function [delta_o, delta_h] = backwardPass( hout, out, y, W2, nodes)
    delta_o = (out-y) .* ((1 + out) .* (1- out)) * 0.5; 
    delta_h = (transpose(W2) * delta_o) .* ((1 + hout) .* (1-hout)) * 0.5;
    delta_h = delta_h(1:nodes, :);
end

function [W1, W2] = weightUpdate(X, W1, W2 ,delta_o , delta_h ,hout,alpha, eta,n, dW1, dW2)
    
  
    dW1 = (dW1 .* alpha) - (delta_h * X') .* (1-alpha);   % kanske fel 
    dW2 = (dW2 .* alpha) - (delta_o * hout') .* (1-alpha);

    W1 = W1 + dW1 .*eta;
    W2 = W2 + dW2 .*eta;
end

function [W1Star, W2Star] = MLP(X, X_test, y, y_test, W1, W2, nodes, n, epochs, alpha, eta, dW1, dW2, n_test)

    lossData = zeros(epochs, 1);
    lossData_val = zeros(epochs, 1);


    accuracyData = zeros(epochs, 1);
    accuracyData_val = zeros(epochs, 1);

    for epoch = 1:epochs
        [hin, hout, oin, out] = forwardPass(X, W1, W2 , n);
        [delta_o, delta_h] = backwardPass( hout, out, y, W2, nodes);
        [W1, W2] = weightUpdate(X, W1, W2 , delta_o, delta_h, hout, alpha, eta,n, dW1, dW2);

        lossData(epoch) = computeCost(y, out) / length(y) * 100;

        [hin_test, hout_test, oin_test, out_test] = forwardPass(X_test, W1, W2 , 100-n);
        lossData_val(epoch)= computeCost(y_test, out_test) / length(y_test) * 100;

        accuracyData(epoch) = (sum(sign(out) == y) / length(y)) * 100;
        accuracyData_val(epoch) = (sum(sign(out_test) == y_test) / length(y_test)) * 100;


        
    end
    plot(lossData, Color = 'Green');
    hold on;
    plot(lossData_val, Color = 'Blue');
    xlabel('Epochs');
    ylabel('MSE');
    title('Mean Squared Error between Train and Val sets');
    legend('Train', 'Val');
    hold off;
    W1Star = W1;
    W2Star = W2;
end

function [classA, classB] = generateClasses(n, mA, mB, sigmaA, sigmaB, non_linear_data, ratioA, ratioB) 
    

        % Determine the number of training samples
        numTrainA = round(ratioA * n);
        numTrainB = round(ratioB * n);

    if non_linear_data
        % Generate non-linear random samples for Class A
        classA(1,:) = [ randn(1,round(0.5*n)) .* sigmaA - mA(1),randn(1,round(0.5*n)) .* sigmaA + mA(1)];
        classA(2,:) = randn(1,n) .* sigmaA + mA(2);
        % Generate non-linear random samples for Class A
        classB(1,:) = randn(1,n) .* sigmaB + mB(1);
        classB(2,:) = randn(1,n) .* sigmaB + mB(2);
    else    
        % Generate random samples for Class A
        classA(1,:) = randn(1,n) .* sigmaA + mA(1);
        classA(2,:) = randn(1,n) .* sigmaA + mA(2);
        
        % Generate random samples for Class B
        classB(1,:) = randn(1,n) .* sigmaB + mB(1);
        classB(2,:) = randn(1,n) .* sigmaB + mB(2);
    end

    % Class Labels
    classA(3,:) = 1;
    classB(3,:) = -1;

end

function [X, y, W1, W2, dW1, dW2] = generateData(classA, classB, n, nodes)
    AandB = cat(2, classA, classB);
    data = AandB(:, randperm(2*n, 2*n));

    % X = Input %
    X = data(1:2, :);
    % y = Labels %
    y = data(3,:);
    % W = Weights
    W1 = randn(nodes, size(X, 1));
    W2 = randn(1, nodes);

    dW1 = zeros(size(W1));
    dW2 = zeros(size(W2));
  

end

function plotData(X, y, W1, W2, classA, classB, out)
    

    figure; 

    scatter(classA(1,:), classA(2,:), 'r', 'filled'); 
    hold on; 
    
    % Plot Class B
    scatter(classB(1,:), classB(2,:), 'b', 'filled'); 
    
    % Labels and title
    xlabel('x1');
    ylabel('x2');
    title('Plot of Class A and Class B');
    legend('Class A', 'Class B');
    grid on; 
    
    axis equal;
    hold off;
end
