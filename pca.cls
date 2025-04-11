\NeedsTeXFormat{LaTeX2e}[1995/12/01]

\ProvidesClass{thesis}[2009/11/15 My Custom latex class]

%%% Load the report class as a base %%%
\LoadClass[b5paper,12pt,twoside]{book}

%%% General packages required %%%
\RequirePackage{calc}

% the main font
\RequirePackage{charter}

%%% Style the headers
\RequirePackage{fancyhdr}
\RequirePackage{color}
\RequirePackage{xcolor}


% this is needed for fancyhdr to work
\setlength{\headheight}{15.2pt}
% define the 'fancy' page style
%% call once to set up the defaults
\pagestyle{fancy}
%% What to put in the boxes
\fancyhead[EL]{{\fontfamily{bch}\selectfont\makebox[27pt][l]{\thepage}\leftmark}}
\fancyhead[OR]{{\fontfamily{bch}\selectfont\rightmark\makebox[27pt][r]{\thepage}}}
\fancyhead[OL]{}
\fancyhead[ER]{}
\fancyfoot[C]{}
%% no line
\renewcommand{\headrulewidth}{0pt}

%% How to display chapter titles in the heading 
\renewcommand{\chaptermark}[1] {
  \markboth{
      \MakeUppercase{\@chapapp\ \thechapter---#1}
  }{}
}

%%% Chapter/section styling %%%
\makeatletter

\renewcommand{\chapter} {
    \clearpage     			% Starts new page.
    \thispagestyle{empty}          	% Page style of chapter page is 'plain'
    \global\@topnum\z@              	% Prevents figures from going at top of page.
    \@afterindentfalse              	% Inserts indent in first paragraph.  Change
    \secdef\@chapter\@schapter      	% to \@afterindentfalse to remove indent.
}

\newlength{\chapnint}

\renewcommand{\@makechapterhead}[1]
{
  \vspace*{12 pt}
  {
    \setlength{\parindent}{0pt} 
    \setlength{\chapnint}{-36pt - 1ex - 1pt} 	% set the negative indent to align the title 
    \hspace{\chapnint}				% with the section numbers
    \raggedright\normalfont\huge \bfseries\scshape
    {\rm\scshape\@chapapp\ \thechapter\thinspace\textperiodcentered\ }#1
    \par\nobreak\vspace{4 pt}
  }
}

\renewcommand{\@makeschapterhead}[1]
{
  \vspace*{12 pt}
  {
    \setlength{\parindent}{0pt} 
    \setlength{\chapnint}{-36pt - 1ex - 1pt} 	% set the negative indent to align the title 
    \hspace{\chapnint}				% with the section numbers
    \raggedright\normalfont\huge\scshape
    {\bfseries#1}
    \par\nobreak\vspace{24 pt}
  }
}

\renewcommand{\section}{\@startsection{section}{1}{0pt}
      {6pt}
      {1pt}{\normalfont\large\bf}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{0pt}
      {3pt}
      {0.1pt}{\normalfont\normalsize\bf}}

\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{0pt}
      {0pt}
      {0.1pt}{\normalfont\normalsize\bf}}

\def\@seccntformat#1{\rlap{\rm\hskip-36pt\csname the#1\endcsname}}


%%% Proof styling %%%


\definecolor{proofcolor}{RGB}{0,0,0}

% Dark orange and Dark Red rgb
\definecolor{theorembordercolor}{RGB}{151, 63, 5}
\definecolor{theorembackgroundcolor}{RGB}{248, 241, 234}

%\newtheoremstyle{theorem}
%{0pt}{0pt}{\normalfont}{0pt}
%{}{\;}{0.25em}
%{{\sffamily\bfseries\color{theorembordercolor}\thmname{#1}~\thmnumber{\textup{#2}}.}
%\thmnote{\normalfont\color{black}~(#3)}}

%%% Helpful environments %%%

\newenvironment{narrow}[2]{%
	\begin{list}{}{%
		\setlength{\topsep}{0pt}%
		\setlength{\leftmargin}{#1}%
		\setlength{\rightmargin}{#2}%
		\setlength{\listparindent}{\parindent}%
		\setlength{\itemindent}{\parindent}%
		\setlength{\parsep}{\parskip}}%
		\item[]}
	{\end{list}}

\newlength{\summarywidth}
\setlength{\summarywidth}{\textwidth - 12pt}

\newenvironment{summary}
{
    \hspace{12pt}
    \begin{minipage}[t]{\summarywidth}
    \begin{it} 
}
{
    \end{it}
    \end{minipage}
    \vspace{0pt}
}

\newcommand{\footnoteremember}[2]{
  \footnote{#2}
  \newcounter{#1}
  \setcounter{#1}{\value{footnote}}
}
\newcommand{\footnoterecall}[1]{
  \footnotemark[\value{#1}]
}