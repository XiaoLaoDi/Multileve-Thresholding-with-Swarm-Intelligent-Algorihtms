function  SFLA_result=SFLA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function SFLA_result=SFLA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The SFLA,proposed by Eusuff and Lansey in 2003,is try to minimize some optimization problems
%    calculating the maximum Kapur-Entropy/Otsu value use the Shuffled Frog Leaping Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    SFLA_result: is a 'Struct' containing the following result information
%        SFLA_result.Fitness:
%            SFLA_result.Fitness.Mean
%            SFLA_result.Fitness.Variance
%            SFLA_result.Fitness.Max
%            SFLA_result.Fitness.Min
%        SFLA_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        SFLA_result.BestThresholds: a vector containing 'TH_num' values
%        SFLA_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        SFLA_result.MeanConvergenceTime: the mean convergence time

disp('the Shuffled Frog Leaping Algorithm is start running...')

%% Shuffled Frog Leaping Algorithm STEPS
    %% STEP ONE
        % Initialize the SFLA parameters
            global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
            fitness=FunctionName;
            T=MAX_Iterations;                                                   % The number of maximum Iteration
            %///////////the following parameters recommended by paper: 
            %           "Multilevel image thresholding by using the shuffled frog-leaping optimization algorithm" T=100;N=50;NG=5;LT=50;
            %           "Improved Shuffled Frog Leaping Algorithm for Continuous Optimization Problem" T=100;N=100;NG=10;LT=50;
            %           "Color Image Segmentation using Clonal Selection-based Shuffled Frog Leaping Algorithm" T=100;N=25;NG=5;LT=20
            %           "The Chaos-based Shuffled Frog Leaping Algorithm and Its Application" T=500;N=200;NG=20;LT=10;
            N=40;                                                   % number of frogs
            NG=8;                                                   % number of groups/memeplex
            LT=10;                                                  % number of iterations in Local Search
            %/////////////////////////////////////////////////////////////////////////////////////////////////////////////
            D_max=0.05*nd;                                          % the maximum Move-Distance
            D=TH_num;                                               % the number of thresholds

%% 预分配内存
SFLA_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
SFLA_result.EachRunBestFitness=zeros(1,RunTimes);
SFLA_result.BestThresholds=zeros(1,TH_num);
SFLA_result.EachRunConvergenceTime=zeros(1,RunTimes);
SFLA_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
SFLA_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
SFLA_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);
%% 最外层循环，让SFLA跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);    
    
    %% STEP TWO
        % Random initialize the Shuffled Frog Leaping Algorithm solutions
            x=zeros(N,D);
            for i=1:N
                for j=1:D
                    x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));           % randomized position
                end
            end
        % Calculate the fitness of each frog 
            Frog_fitness=zeros(1,N);
            for i=1:N    
                Frog_fitness(i)=1/fitness(LP,x(i,:));
                FunCount=FunCount+1;
            end
    %% STEP THREE
        %% the SFLA main iterations
            Each_Iterate_BestFitness=zeros(1,T);                    % the optimal fitness of each iteration
            Each_Iterate_BestThresholds=zeros(T,D);                 % the optimal frog of each iteration
            Success_FindBest_Index(RepeatTimes)=0;
            Iterates=1;
            SUCESS_FLAG=0;
            SUCESS_FLAG_R=0;
             while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T) 
                % Sorting the Frog fitness and Grouping
                    % Sorting    
                    [fitsort,sortindex]=sort(Frog_fitness);
                    x=x(sortindex,:);
                    Each_Iterate_BestFitness(Iterates)=fitsort(1);
                    Each_Iterate_BestThresholds(Iterates,:)=x(1,:);
                    if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                        SFLA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                        EachRunFunCalNum(Iterates)=FunCount;
                        Success_FindBest_Index(RepeatTimes)=1;
                        break;
                    end
                    % Grouping
                    for i=1:NG
                        location=x(i:NG:end,:);
                        fitlocal=fitsort(i:NG:end);
                        % local search
                        for j=1:LT
                            F_b=location(1,:);
                            F_w=location(N/NG,:);
                            % update the worst Frog
                                Move_Dis=(2*rand(1,D)-1).*(F_b-F_w);
                                Move_Dis(Move_Dis>D_max)=D_max;
                                Move_Dis(Move_Dis<-D_max)=-D_max;
                                new_F_w=F_w+Move_Dis;                           % updating equation
                                new_F_w=round(new_F_w);
                                new_F_w(new_F_w>nd)=nd;
                                new_F_w(new_F_w<st)=st;
                            % if the worst frog's fitness is not improved then execute the following update equation
                                CompTemp=fitness(LP,new_F_w);
                                UpdatedFwFitness=1/CompTemp;
                                FunCount=FunCount+1;
                                if UpdatedFwFitness > fitlocal(N/NG)
                                    Move_Dis=(2*rand(1,D)-1).*(Each_Iterate_BestThresholds(Iterates,:)-F_w);
                                    Move_Dis(Move_Dis>D_max)=D_max;
                                    Move_Dis(Move_Dis<-D_max)=-D_max;
                                    new_F_w=F_w+Move_Dis;                       % updating equation
                                    new_F_w=round(new_F_w);
                                    new_F_w(new_F_w>nd)=nd;
                                    new_F_w(new_F_w<st)=st;
                                    % after the two up updating rules,if the worst frog's fitness still not improved then execute the following random operator
                                        CompTemp=fitness(LP,new_F_w);
                                        UpdatedFwFitness=1/CompTemp;
                                        FunCount=FunCount+1;
                                        if UpdatedFwFitness > fitlocal(N/NG)
                                            new_F_w=st+round((nd-st).*rand(1,D));
                                            CompTemp=fitness(LP,new_F_w);
                                            UpdatedFwFitness=1/CompTemp;
                                            FunCount=FunCount+1;
                                        end
                                end
                            % update the worst frog
                                location(N/NG,:)=new_F_w;
                                fitlocal(N/NG)=UpdatedFwFitness;
                                if(fitness(LP,Each_Iterate_BestThresholds(Iterates,:)))<CompTemp
                                    Each_Iterate_BestThresholds(Iterates,:)=new_F_w;
                                    if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                                        SFLA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                                        EachRunFunCalNum(Iterates)=FunCount;
                                        Each_Iterate_BestFitness(Iterates)=fitlocal(N/NG);
                                        SUCESS_FLAG=1;
                                        break;
                                    end
                                end
                            % resorting memeplex's fitness and iterating
                                [fitlocal,indexlocal]=sort(fitlocal);
                                location=location(indexlocal,:);
                        end
                        if(SUCESS_FLAG==1)
                            Success_FindBest_Index(RepeatTimes)=1;
                            SUCESS_FLAG_R=1;
                            break;
                        end
                        x(i:NG:end,:)=location;
                        fitsort(i:NG:end)=fitlocal;
                        Frog_fitness(i:NG:end)=fitsort(i:NG:end);
                    end
                    if SUCESS_FLAG_R==1
                        break;
                    end
                SFLA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                EachRunFunCalNum(Iterates)=FunCount;              
                Iterates=Iterates+1;
            end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"
%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    SFLA_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    SFLA_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    SFLA_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    SFLA_result.EachRunConvergenceTime(RepeatTimes)=SFLA_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    SFLA_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"

    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [SFLA_result.Fitness.Max,SFLAFMax_Index]=max(SFLA_result.EachRunBestFitness);
    [SFLA_result.Fitness.Min,~]=min(SFLA_result.EachRunBestFitness);
    SFLA_result.Fitness.Mean=mean(SFLA_result.EachRunBestFitness);
    SFLA_result.Fitness.Variance=var(SFLA_result.EachRunBestFitness);
    SFLA_result.BestThresholds=sort(SFLA_result.EachRunBestThresholds(SFLAFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    SFLA_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    SFLA_result.MeanConvergenceTime=mean(SFLA_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存SFLA结果：ImageNmae_THChar_SFLA_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_SFLA_result.mat');
    save(FILENAME,'SFLA_result');      
        
disp('the Shuffled Frog Leaping Algorithm is accomplised !!!')

end