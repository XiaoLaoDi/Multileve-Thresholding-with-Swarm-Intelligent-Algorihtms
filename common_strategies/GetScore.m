function Score=GetScore(A)
% function Score=GetScore(A)
% Input A: A sequence need to sort
% Output Score: the sort index
% eg:
%  if A=[1 2 4 1 5]
%  then Score=[1 2 3 1 4]

SortA=sort(A);
N=length(A);
ScoreA=zeros(1,N);
for i=1:N
    for j=1:N
        if A(i)==SortA(j)
            ScoreA(i)=j;
            break;
        end
    end
end
Score=zeros(1,N);
ScoreCnt=1;
for i=1:N
    Tmp=min(ScoreA);
    if Tmp==inf
        break;
    end
    Index=find(ScoreA==Tmp);
    SameScoreLength=length(Index);
    Score(Index)=ScoreCnt;
    ScoreA(Index)=inf;
    ScoreCnt=ScoreCnt+SameScoreLength;
end

end
