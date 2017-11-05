function  PatternTypeTwo_CS_result=PatternTypeTwo_CS_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function PatternTypeTwo_CS_result=PatternTypeTwo_CS_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The Cuckoo Search,proposed by Xin-She Yang and Suash Deb in 2009,is try to optimize some minimization problems
%    Calculating the maximum Kapur-Entropy/Otsu value use the Firefly Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    PatternTypeTwo_CS_result: is a 'Struct' containing the following result information
%        PatternTypeTwo_CS_result.Fitness:
%            PatternTypeTwo_CS_result.Fitness.Mean
%            PatternTypeTwo_CS_result.Fitness.Variance
%            PatternTypeTwo_CS_result.Fitness.Max
%            PatternTypeTwo_CS_result.Fitness.Min
%        PatternTypeTwo_CS_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        PatternTypeTwo_CS_result.BestThresholds: a vector containing 'TH_num' values
%        PatternTypeTwo_CS_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        PatternTypeTwo_CS_result.MeanConvergenceTime: the mean convergence time
disp('the Pattern Type Two Cuckoo Search is running...')

%% Cuckoo Search STEPS
    %% STEP ONE
        % Initialize the PatternTypeTwo_CS parameters
        global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
            fitness=FunctionName;
            N=20;                                                   % Number of nests (or different solutions)
            Pa=0.25;                                                % Discovery rate of alien eggs/solutions
            T=MAX_Iterations;                                       % Maximum iterations
            D=TH_num;                                               % the number of thresholds
PS_SUCCESS_FLAG=0;                        

%% 预分配内存
PatternTypeTwo_CS_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
PatternTypeTwo_CS_result.EachRunBestFitness=zeros(1,RunTimes);
PatternTypeTwo_CS_result.BestThresholds=zeros(1,TH_num);
PatternTypeTwo_CS_result.EachRunConvergenceTime=zeros(1,RunTimes);
PatternTypeTwo_CS_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
PatternTypeTwo_CS_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
PatternTypeTwo_CS_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让PatternTypeTwo_CS跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');  
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
    %% STEP TWO
        % Random initialize the Cuckoo Search solutions
            nest=zeros(N,D);
            for i=1:N
                for j=1:D
                    nest(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));       % randomized position
                end
            end
        % calculate fitness , get the current best solution(i.e. bestnest) & fitness(i.e.bestfitness)
            nestfitness=zeros(1,N);
            for i=1:N
                nestfitness(i)=1/fitness(LP,nest(i,:));
                FunCount=FunCount+1;
            end
    %% STEP THREE
        %% the PatternTypeTwo_CS main iterations
            Each_Iterate_BestFitness=zeros(1,T);                    % the optimal fitness of each iteration
            Each_Iterate_BestThresholds=zeros(T,D);                 % the optimal solution of each iteration
            Iterates=1;
            Success_FindBest_Index(RepeatTimes)=0;
            while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)                 
                 %% STEP FOUR
                        [Each_Iterate_BestFitness(Iterates),index]=min(nestfitness);
                        Each_Iterate_BestThresholds(Iterates,:)=nest(index,:);
                        if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                            PatternTypeTwo_CS_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
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
        PatternTypeTwo_CS_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Success_FindBest_Index(RepeatTimes)=1;
        break;
     end
 end
                        
                %% (1)Generate new solutions (but keep the current best) by Levy flights
                        beta=3/2;           % Levy exponent and coefficient,For details, see equation (2.21), 
                                            % Page 16 (chapter 2) of the book "X. S. Yang, Nature-Inspired 
                                            % Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010)."
                        sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
                        new_nest=zeros(N,D);
                        for i=1:N   
                            % This is a simple way of implementing Levy flights,For standard random walks, use step=1;
                            % Levy flights by Mantegna's algorithm
                                u=randn(1,D)*sigma;
                                v=randn(1,D);
                                step=u./abs(v).^(1/beta);
                            % In the next equation, the difference factor (nest-bestnest) means that when the
                            % solution is the best solution, it remains unchanged.     
                                stepsize=0.01*step.*(nest(i,:)-Each_Iterate_BestThresholds(Iterates,:));    % Here the factor 0.01 comes from the fact 
                                                                                                            % that L/100 should the typical step size of 
                                                                                                            % walks/flights where L is the typical lenghtscale; 
                                                                                                            % otherwise, Levy flights may become too aggresive/efficient,
                                                                                                            % which makes new solutions (even) jump out side of the design 
                                                                                                            % domain (and thus wasting evaluations).
                            % Now the actual random walks or flights
                                new_nest(i,:)=nest(i,:)+stepsize.*randn(1,D);
                                new_nest(i,:)=round(new_nest(i,:));
                           % Apply simple bounds/limits
                               new_nest(new_nest>nd)=nd;
                               new_nest(new_nest<st)=st;
                        end
                %% （2）calculate objected-fitness , get the solution & fitness after Levy flights
                        new_nestfitness=zeros(1,N);
                        for i=1:N
                            new_nestfitness(i)=1/fitness(LP,new_nest(i,:));
                            FunCount=FunCount+1;
                            if new_nestfitness(i) < nestfitness(i)
                                nest(i,:)=new_nest(i,:);
                                nestfitness(i)=new_nestfitness(i);
                            end
                        end
                %% （4）Discovery and randomization
                        % A fraction of worse nests are discovered with a probability Pa
                        % Discovered or not -- a status vector
                            K=rand(N,D)>Pa;
                        % New solution by biased/selective random walks
                            stepsize=rand*(nest(randperm(N),:)-nest(randperm(N),:));    % In the real world, if a cuckoo's egg is very similar 
                                                                                        % to a host's eggs, then this cuckoo's egg is less likely 
                                                                                        % to be discovered, thus the fitness should be related to 
                                                                                        % the difference in solutions.  Therefore, it is a good idea 
                                                                                        % to do a random walk in a biased way with some random step sizes.  
                            new_nest=nest+stepsize.*K;
                            new_nest=round(new_nest);
                        % Apply simple bounds/limits
                            new_nest(new_nest>nd)=nd;
                            new_nest(new_nest<st)=st;
                %% （5）Evaluate this set of solutions
                            new_nestfitness=zeros(1,N);
                            for i=1:N
                                new_nestfitness(i)=1/fitness(LP,new_nest(i,:));
                                FunCount=FunCount+1;
                                if new_nestfitness(i) < nestfitness(i)
                                    nest(i,:)=new_nest(i,:);
                                    nestfitness(i)=new_nestfitness(i);
                                end
                            end
                PatternTypeTwo_CS_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;
                Iterates=Iterates+1;
            end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    PatternTypeTwo_CS_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    PatternTypeTwo_CS_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    PatternTypeTwo_CS_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    PatternTypeTwo_CS_result.EachRunConvergenceTime(RepeatTimes)=PatternTypeTwo_CS_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    PatternTypeTwo_CS_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [PatternTypeTwo_CS_result.Fitness.Max,PatternTypeTwo_CSFMax_Index]=max(PatternTypeTwo_CS_result.EachRunBestFitness);
    [PatternTypeTwo_CS_result.Fitness.Min,~]=min(PatternTypeTwo_CS_result.EachRunBestFitness);
    PatternTypeTwo_CS_result.Fitness.Mean=mean(PatternTypeTwo_CS_result.EachRunBestFitness);
    PatternTypeTwo_CS_result.Fitness.Variance=var(PatternTypeTwo_CS_result.EachRunBestFitness);
    PatternTypeTwo_CS_result.BestThresholds=sort(PatternTypeTwo_CS_result.EachRunBestThresholds(PatternTypeTwo_CSFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    PatternTypeTwo_CS_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    PatternTypeTwo_CS_result.MeanConvergenceTime=mean(PatternTypeTwo_CS_result.EachRunConvergenceTime); 

disp('Statistic Metrics is Calculated !')

% 保存PatternTypeTwo_CS结果：ImageNmae_THChar_PatternTypeTwo_CS_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_PatternTypeTwo_CS_result.mat');
    save(FILENAME,'PatternTypeTwo_CS_result');     
        
disp('the Pattern Type Two Cuckoo Search is accomplised !!!')

end
