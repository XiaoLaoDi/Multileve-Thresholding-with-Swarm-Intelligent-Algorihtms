function  PatternTypeTwo_FA_result=PatternTypeTwo_FA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function PatternTypeTwo_FA_result=PatternTypeTwo_FA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    Firefly Algorithm (FA) developed by Xin-she Yang at Cambridge University in 2008 in the Chapter 8 of 'monograph ONE'
%    Yang,X.S.:Nature-inspired metaheuristic algorithms,Luniver Press,(2008)--refered as 'Monograph ONE'
%    Szymon Lukasik and Slawomir Zak in "Firefly Algorithm for Continuous Constrained Optimization Tasks"---refered as ‘Paper TWO’
%    Note: the MFA is to minimize some optimization problem
%    Calculating the maximum Kapur-Entropy/Otsu value use the Firefly Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    PatternTypeTwo_FA_result: is a 'Struct' containing the following result information
%        PatternTypeTwo_FA_result.Fitness:
%            PatternTypeTwo_FA_result.Fitness.Mean
%            PatternTypeTwo_FA_result.Fitness.Variance
%            PatternTypeTwo_FA_result.Fitness.Max
%            PatternTypeTwo_FA_result.Fitness.Min
%        PatternTypeTwo_FA_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        PatternTypeTwo_FA_result.BestThresholds: a vector containing 'TH_num' values
%        PatternTypeTwo_FA_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        PatternTypeTwo_FA_result.MeanConvergenceTime: the mean convergence time

disp('the Pattern Type Two Firefly Algorithm is running...')

%% Pre-processing: Initializing
% STEP ONE
    % Initialize the problem parameters
        global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
        fitness=FunctionName;                            % fitness: the problem which required to maximize 
        D=TH_num;                                           % D: the Dimension of the problem
        x_max=nd;                                           % x_max: the up-boundary of each Dimension
        x_min=st;                                           % x_min: the down-limit of each Dimension 
        scale=x_max-x_min;
% STEP TWO
    % Initialize PatternTypeTwo_FA parameters
    % the initialized PatternTypeTwo_FA parameters suggested by 'Paper TWO'
        alpha=0.006;                                         %α∈[0,1]
        betamax=0.8;                                        % β0 is the maximum attractiveness value
        gamma=1;
        Num=40;                                             % Num: number of fireflies
        T=MAX_Iterations;                                   % T: maximum iterations

PS_SUCCESS_FLAG=0;                        
        
%% 预分配内存
PatternTypeTwo_FA_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
PatternTypeTwo_FA_result.EachRunBestFitness=zeros(1,RunTimes);
PatternTypeTwo_FA_result.BestThresholds=zeros(1,TH_num);
PatternTypeTwo_FA_result.EachRunConvergenceTime=zeros(1,RunTimes);
PatternTypeTwo_FA_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
PatternTypeTwo_FA_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
PatternTypeTwo_FA_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让PatternTypeTwo_FA跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');  
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
% STEP THREE
    % Initialize the Firefly 
        x=zeros(Num,D); 
        for i=1:Num
            for j=1:D
                x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));       % randomized position
            end
        end 
    % Calculate the absolute light intensity of the firefly swarm   
         ABS_brightness=zeros(1,Num);
         
%% Main iterations of firefly algorithm
% Main loops
    Each_Iterate_BestFitness=zeros(1,T);                    % the optimal fitness of each iteration
    Each_Iterate_BestThresholds=zeros(T,D);                 % the optimal solution of each iteration
    Iterates=1;
    Success_FindBest_Index(RepeatTimes)=0;
     while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)   
        % Find the best solution index
            for i=1:Num
                ABS_brightness(i)=1/fitness(LP,x(i,:));
                FunCount=FunCount+1;
            end
            [ABS_brightnessSorted,index]=sort(ABS_brightness);
            ABS_brightness=ABS_brightnessSorted;
            x=x(index,:);
            Each_Iterate_BestFitness(Iterates)=ABS_brightness(1);
            Each_Iterate_BestThresholds(Iterates,:)=x(1,:);
            if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                PatternTypeTwo_FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
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
        PatternTypeTwo_FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Success_FindBest_Index(RepeatTimes)=1;
        break;
     end
 end

            
            
            
        % Inner loops of PatternTypeTwo_FA
        for i=1:Num
            for j=1:i
                if ABS_brightness(j) < ABS_brightness(i)
                % Calculate the distance between Firefly i and Firefly j
                    r=sqrt(sum((x(i,:)-x(j,:)).^2));
                % Calculate the attractiveness of Firefly i to Firefly j
                    beta=betamax*exp(-gamma*r.^2)+(1-betamax);
                % Firefly i move to Firefly j
                    % MFA(Modified Firefly Algorithm) iterative equatio
                    x(i,:)=x(i,:)+beta*(x(j,:)-x(i,:))+alpha*scale.*(rand(1,D)-0.5); 
                    x(i,:)=round(x(i,:));
                    x(x>nd)=nd;
                    x(x<st)=st;
                end
            end
        end
         PatternTypeTwo_FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
         EachRunFunCalNum(Iterates)=FunCount;
         Iterates=Iterates+1;
     end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    PatternTypeTwo_FA_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    PatternTypeTwo_FA_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    PatternTypeTwo_FA_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    PatternTypeTwo_FA_result.EachRunConvergenceTime(RepeatTimes)=PatternTypeTwo_FA_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    PatternTypeTwo_FA_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);

end % End "for RepeatTimes=1:RunTimes"

    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [PatternTypeTwo_FA_result.Fitness.Max,PatternTypeTwo_FAFMax_Index]=max(PatternTypeTwo_FA_result.EachRunBestFitness);
    [PatternTypeTwo_FA_result.Fitness.Min,~]=min(PatternTypeTwo_FA_result.EachRunBestFitness);
    PatternTypeTwo_FA_result.Fitness.Mean=mean(PatternTypeTwo_FA_result.EachRunBestFitness);
    PatternTypeTwo_FA_result.Fitness.Variance=var(PatternTypeTwo_FA_result.EachRunBestFitness);
    PatternTypeTwo_FA_result.BestThresholds=sort(PatternTypeTwo_FA_result.EachRunBestThresholds(PatternTypeTwo_FAFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    PatternTypeTwo_FA_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    PatternTypeTwo_FA_result.MeanConvergenceTime=mean(PatternTypeTwo_FA_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存PatternTypeTwo_FA结果：ImageNmae_THChar_PatternTypeTwo_FA_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_PatternTypeTwo_FA_result.mat');
    save(FILENAME,'PatternTypeTwo_FA_result');      

disp('the Pattern Type Two Firefly Algorithm is accomplised !!!')
end
