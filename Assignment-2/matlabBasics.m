%Basic Data Element - Matrix

%Matrices%

%Row Matrix
A = [5 4 3 2 1]
%Column Matrix
B = [1; 2; 3; 4; 5;]

%To clear command window
clc
%To clear workspace
clear all

%Row by Column Matrix
C = [1 2 3;4 5 6;7 8 9]

det(C) %determinant
inv(C) %inverse
C' %transpose

%Index starts from 1 NOT 0
A = [1 2 3 4]
A(1)
A(2:4) % 2,3,4 elements
A(2:end) % 2 till end
C
C(:,2)

size(C) %order of matrix
C*C %matrix multiplication

ones(3,3)
zeros(3,3)


D = [1 2 3]
E = [1; 2; 3;]
D*E