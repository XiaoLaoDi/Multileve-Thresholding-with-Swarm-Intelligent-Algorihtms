function  PatternTypeTwo_ALO_result=PatternTypeTwo_ALO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function PatternTypeTwo_ALO_result=PatternTypeTwo_ALO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The Ant Lion Optimizer,proposed by S. Mirjalili, 
%    The Ant Lion Optimizer,Advances in Engineering Software , in press,2015. DOI: http://dx.doi.org/10.1016/j.advengsoft.2015.01.010 
%    Calculating the maximum Kapur_Entropy/Between-Variance use the Ant Lion Optimizer
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    PatternTypeTwo_ALO_result: is a 'Struct' containing the following result information
%        PatternTypeTwo_ALO_result.Fitness:
%            PatternTypeTwo_ALO_result.Fitness.Mean
%            PatternTypeTwo_ALO_result.Fitness.Variance
%            PatternTypeTwo_ALO_result.Fitness.Max
%            PatternTypeTwo_ALO_result.Fitness.Min
%        PatternTypeTwo_ALO_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        PatternTypeTwo_ALO_result.BestThresholds: a vector containing 'TH_num' values
%        PatternTypeTwo_ALO_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        PatternTypeTwo_ALO_result.MeanConvergenceTime: the mean convergence time

disp('the Pattern Type Two Ant Lion Optimizer is running...')

%% PatternTypeTwo_ALO STEPS
% STEP ONE
% Initialize the PatternTypeTwo_ALO parameters
global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
fitness=FunctionName;
T=MAX_Iterations;
D=TH_num;
N=40;

%% 预分配内存
%记录结果
PatternTypeTwo_ALO_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
PatternTypeTwo_ALO_result.EachRunBestFitness=zeros(1,RunTimes);
PatternTypeTwo_ALO_result.BestThresholds=zeros(1,TH_num);
PatternTypeTwo_ALO_result.EachRunConvergenceTime=zeros(1,RunTimes);
PatternTypeTwo_ALO_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
PatternTypeTwo_ALO_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
PatternTypeTwo_ALO_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让PatternTypeTwo_ALO跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    % STEP TWO
    %------Initialize the positions of antlions and ants------------
        antlion_position=zeros(N,D);
        ant_position=zeros(N,D);
        for i=1:N
            for j=1:D
                antlion_position(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));            %randomized position 
                ant_position(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));                %randomized position
            end
        end    
% Initialize variables to save the position of elite, sorted antlions, 
% convergence curve, antlions fitness, and ants fitness
    Sorted_antlions=zeros(N,D);
    antlions_fitness=zeros(1,N);
    ants_fitness=zeros(1,N);
% Calculate the fitness of initial antlions and sort them
    for i=1:N
        antlions_fitness(1,i)=1/fitness(LP,antlion_position(i,:)); 
        FunCount=FunCount+1;
    end
    [sorted_antlion_fitness,sorted_indexes]=sort(antlions_fitness);    
    for newindex=1:N
         Sorted_antlions(newindex,:)=antlion_position(sorted_indexes(newindex),:);
    end    
    Elite_antlion_position=Sorted_antlions(1,:);
    Elite_antlion_fitness=sorted_antlion_fitness(1);

% STEP THREE
%------the PatternTypeTwo_ALO main iterations ------------
    Each_Iterate_BestFitness=zeros(1,T);
    Each_Iterate_BestThresholds=zeros(T,D);
    Iterates=1;
    Success_FindBest_Index(RepeatTimes)=0;
    while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)        
        % iterations
            Each_Iterate_BestFitness(Iterates)=Elite_antlion_fitness;
            Each_Iterate_BestThresholds(Iterates,:)=Elite_antlion_position;
            if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                PatternTypeTwo_ALO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;
                Success_FindBest_Index(RepeatTimes)=1;
                break;
            end           
            
X_old=Each_Iterate_BestThresholds(Iterates,:);
fitness_old=Each_Iterate_BestFitness(Iterates);
[X_new,Fitness_new,PS_SUCCESS_FLAG,FunCount]=MyPatternSearch(fitness,X_old,fitness_old,D,FunCount);
 if PS_SUCCESS_FLAG==1
     Each_Iterate_BestThresholds(Iterates,:)=X_new;
        Each_Iterate_BestFitness(Iterates)=Fitness_new;
     if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
        PatternTypeTwo_ALO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Success_FindBest_Index(RepeatTimes)=1;
        break;
     end
 end          
        
        % This for loop simulate random walks
            for i=1:N
                % Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
                    Rolette_index=RouletteWheelSelection(1./sorted_antlion_fitness);
                    if Rolette_index==-1  
                        Rolette_index=1;
                    end     
                % RA is the random walk around the selected antlion by rolette wheel
                    RA=Random_walk_around_antlion(D,MAX_Iterations,st,nd, Sorted_antlions(Rolette_index,:),Iterates);       
                % RA is the random walk around the elite (best antlion so far)
                    RE=Random_walk_around_antlion(D,MAX_Iterations,st,nd, Elite_antlion_position(1,:),Iterates);     
                    ant_position(i,:)= (RA(Iterates,:)+RE(Iterates,:))/2; % Equation (2.13) in the paper          
            end   
            for i=1:N       
                % Boundar checking (bring back the antlions of ants inside search
                % space if they go beyoud the boundaries
                    Flag4ub=ant_position(i,:)>nd;
                    Flag4lb=ant_position(i,:)<st;
                    ant_position(i,:)=(ant_position(i,:).*(~(Flag4ub+Flag4lb)))+nd.*Flag4ub+st.*Flag4lb; 
                    ant_position(i,:)=round(ant_position(i,:));
                    ants_fitness(1,i)=1/fitness(LP,ant_position(i,:)); 
                    FunCount=FunCount+1;
            end    
        % Update antlion positions and fitnesses based of the ants (if an ant becomes fitter than an antlion we assume
        % it was cought by the antlion and the antlion update goes to its position to build the trap)
            double_population=[Sorted_antlions;ant_position];
            double_fitness=[sorted_antlion_fitness ants_fitness];        
            [double_fitness_sorted I]=sort(double_fitness);
            double_sorted_population=double_population(I,:);       
            antlions_fitness=double_fitness_sorted(1:N);
            Sorted_antlions=double_sorted_population(1:N,:);       
        % Update the position of elite if any antlinons becomes fitter than it
            if antlions_fitness(1)<Elite_antlion_fitness 
                Elite_antlion_position=Sorted_antlions(1,:);
                Elite_antlion_fitness=antlions_fitness(1);
            end     
        % Keep the elite in the population
            Sorted_antlions(1,:)=Elite_antlion_position;
            antlions_fitness(1)=Elite_antlion_fitness;    
        PatternTypeTwo_ALO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Iterates=Iterates+1;
    end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"   
    
%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    PatternTypeTwo_ALO_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    PatternTypeTwo_ALO_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    PatternTypeTwo_ALO_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    PatternTypeTwo_ALO_result.EachRunConvergenceTime(RepeatTimes)=PatternTypeTwo_ALO_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    PatternTypeTwo_ALO_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [PatternTypeTwo_ALO_result.Fitness.Max,PatternTypeTwo_ALOFMax_Index]=max(PatternTypeTwo_ALO_result.EachRunBestFitness);
    [PatternTypeTwo_ALO_result.Fitness.Min,~]=min(PatternTypeTwo_ALO_result.EachRunBestFitness);
    PatternTypeTwo_ALO_result.Fitness.Mean=mean(PatternTypeTwo_ALO_result.EachRunBestFitness);
    PatternTypeTwo_ALO_result.Fitness.Variance=var(PatternTypeTwo_ALO_result.EachRunBestFitness);
    PatternTypeTwo_ALO_result.BestThresholds=sort(PatternTypeTwo_ALO_result.EachRunBestThresholds(PatternTypeTwo_ALOFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    PatternTypeTwo_ALO_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    PatternTypeTwo_ALO_result.MeanConvergenceTime=mean(PatternTypeTwo_ALO_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存PatternTypeTwo_ALO结果：ImageNmae_THChar_PatternTypeTwo_ALO_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_PatternTypeTwo_ALO_result.mat');
    save(FILENAME,'PatternTypeTwo_ALO_result');     

disp('the Pattern Type Two Ant Lion Optimizer is accomplised !!!')



end