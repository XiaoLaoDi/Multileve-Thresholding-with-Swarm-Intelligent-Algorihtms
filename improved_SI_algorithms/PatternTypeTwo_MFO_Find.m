function  PatternTypeTwo_MFO_result=PatternTypeTwo_MFO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function PatternTypeTwo_MFO_result=PatternTypeTwo_MFO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The Moth-Flame Optimization Algorithm (MFO),proposed  S. Mirjalili, 
%    Moth-Flame Optimization Algorithm: A Novel Nature-inspired Heuristic Paradigm, Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.07.006
%    Calculating the maximum Kapur_Entropy/Between-Variance use the Moth-Flame Optimization Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Func_Counts:
% Output:
%    PatternTypeTwo_MFO_result: is a 'Struct' containing the following result information
%        PatternTypeTwo_MFO_result.Fitness:
%            PatternTypeTwo_MFO_result.Fitness.Mean
%            PatternTypeTwo_MFO_result.Fitness.Variance
%            PatternTypeTwo_MFO_result.Fitness.Max
%            PatternTypeTwo_MFO_result.Fitness.Min
%        PatternTypeTwo_MFO_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        PatternTypeTwo_MFO_result.BestThresholds: a vector containing 'TH_num' values
%        PatternTypeTwo_MFO_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        PatternTypeTwo_MFO_result.MeanConvergenceTime: the mean convergence time

disp('the Pattern Type Two Moth-Flame Optimization Algorithm is running...')

%% WOA STEPS
% STEP ONE
% Initialize the MFO parameters
global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;

N=40;
D=TH_num;
fitness=FunctionName;
T=MAX_Iterations;
LinearConstant=4000;

%% 预分配内存
PatternTypeTwo_MFO_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
PatternTypeTwo_MFO_result.EachRunBestFitness=zeros(1,RunTimes);
PatternTypeTwo_MFO_result.BestThresholds=zeros(1,TH_num);
PatternTypeTwo_MFO_result.EachRunConvergenceTime=zeros(1,RunTimes);
PatternTypeTwo_MFO_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
PatternTypeTwo_MFO_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
PatternTypeTwo_MFO_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);

%% 最外层循环，让WOA跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');
    
	tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
    % STEP TWO
    %------Initialize the WOA swarm's position and velocity------------
    %Initialize the positions of search agents
	x=zeros(N,D);
    Moth_fitness=zeros(1,N);
	for i=1:N
		for j=1:D
			x(i,j)=st+round((nd-st)*rand);               % randomized position
		end
	end

    % STEP THREE
    %------the WOA main iterations ------------
    Each_Iterate_BestFitness=zeros(1,T);
    Each_Iterate_BestThresholds=zeros(T,D);
    Iterates=1;
    Success_FindBest_Index(RepeatTimes)=0;
    while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)
        % Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iterates*((N-1)/MAX_Iterations));
        for i=1:N
            % Check if moths go out of the search spaceand bring it back
            Flag4ub=x(i,:)>nd;
            Flag4lb=x(i,:)<st;
            x(i,:)=(x(i,:).*(~(Flag4ub+Flag4lb)))+nd.*Flag4ub+st.*Flag4lb;  
            x(i,:)=round(x(i,:));
            % Calculate the fitness of moths
            Moth_fitness(1,i)=1/fitness(LP,x(i,:)); 
            FunCount=FunCount+1;
        end    

        if Iterates==1
            % Sort the first population of moths
            [fitness_sorted I]=sort(Moth_fitness);
            sorted_population=x(I,:);       
            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=fitness_sorted;
        else       
            % Sort the moths
            double_population=[previous_population;best_flames];
            double_fitness=[previous_fitness best_flame_fitness];

            [double_fitness_sorted I]=sort(double_fitness);
            double_sorted_population=double_population(I,:);

            fitness_sorted=double_fitness_sorted(1:N);
            sorted_population=double_sorted_population(1:N,:);

            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=fitness_sorted;
        end

        % Update the position best flame obtained so far
        Best_flame_score=fitness_sorted(1);
        Best_flame_pos=sorted_population(1,:);

        % iterations
        Each_Iterate_BestFitness(Iterates)=Best_flame_score;
        Each_Iterate_BestThresholds(Iterates,:)=Best_flame_pos;
        if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
            PatternTypeTwo_MFO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
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
        PatternTypeTwo_MFO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Success_FindBest_Index(RepeatTimes)=1;
        break;
     end
 end

        previous_population=x;
        previous_fitness=Moth_fitness;

        % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a=-1+Iterates*((-1)/LinearConstant);
        for i=1:size(x,1)
            for j=1:size(x,2)
                if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame  
                    % D in Eq. (3.13)
                    distance_to_flame=abs(sorted_population(i,j)-x(i,j));
                    b=1;
                    t=(a-1)*rand+1;
                    % Eq. (3.12)
                    x(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(i,j);
                end
                if i>Flame_no % Upaate the position of the moth with respct to one flame
                    % Eq. (3.13)
                    distance_to_flame=abs(sorted_population(i,j)-x(i,j));
                    b=1;
                    t=(a-1)*rand+1;
                    % Eq. (3.12)
                    x(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
                end
            end
        end 

        PatternTypeTwo_MFO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Iterates=Iterates+1;
    end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    PatternTypeTwo_MFO_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    PatternTypeTwo_MFO_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    PatternTypeTwo_MFO_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    PatternTypeTwo_MFO_result.EachRunConvergenceTime(RepeatTimes)=PatternTypeTwo_MFO_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    PatternTypeTwo_MFO_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [PatternTypeTwo_MFO_result.Fitness.Max,PatternTypeTwo_MFOFMax_Index]=max(PatternTypeTwo_MFO_result.EachRunBestFitness);
    [PatternTypeTwo_MFO_result.Fitness.Min,~]=min(PatternTypeTwo_MFO_result.EachRunBestFitness);
    PatternTypeTwo_MFO_result.Fitness.Mean=mean(PatternTypeTwo_MFO_result.EachRunBestFitness);
    PatternTypeTwo_MFO_result.Fitness.Variance=var(PatternTypeTwo_MFO_result.EachRunBestFitness);
    PatternTypeTwo_MFO_result.BestThresholds=sort(PatternTypeTwo_MFO_result.EachRunBestThresholds(PatternTypeTwo_MFOFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    PatternTypeTwo_MFO_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    PatternTypeTwo_MFO_result.MeanConvergenceTime=mean(PatternTypeTwo_MFO_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存WOA结果：ImageNmae_THChar_PatternTypeTwo_MFO_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_PatternTypeTwo_MFO_result.mat');
    save(FILENAME,'PatternTypeTwo_MFO_result');     

disp('the Pattern Type Two Moth-Flame Optimization Algorithm is accomplised !!!')



end