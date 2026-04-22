% llncs.cls  —  Springer LNCS document class (minimal working version)
% Based on the official Springer LNCS class, llncs2e (public domain header).
% For full production use, download the official package from:
%   https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
% and replace this file with the official llncs.cls.
%
% This stub is sufficient to compile and check layout locally.
% ---------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{llncs}[2022/07/07 LLNCS Springer class (stub)]
\LoadClass[10pt,twoside]{article}
\RequirePackage{mathptmx}
\RequirePackage[T1]{fontenc}
\RequirePackage{geometry}
\geometry{a4paper,top=2.2cm,bottom=2.2cm,left=2.2cm,right=2.2cm}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{abstract}

% Running heads
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[LO]{\@runninghead}
\fancyhead[RE]{\@runningtitle}
\newcommand{\@runninghead}{}
\newcommand{\@runningtitle}{}
\newcommand{\titlerunning}[1]{\renewcommand{\@runningtitle}{#1}}
\newcommand{\authorrunning}[1]{\renewcommand{\@runninghead}{#1}}

% Institute
\newcommand{\institute}[1]{\def\@institute{#1}}
\def\@institute{}

% Keywords
\newenvironment{keywords}{\par\vspace{6pt}\noindent\textbf{Keywords: }}{\par}

% Maketitle
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}\vspace{1em}
    {\large\@author\par}\vspace{0.4em}
    {\small\@institute\par}
  \end{center}
  \vspace{1em}
}

% Abstract
\renewenvironment{abstract}{%
  \begin{center}\textbf{Abstract}\end{center}
  \begin{quote}\small
}{%
  \end{quote}
}

% Section formatting (LNCS style)
\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape}{\thesubsubsection}{1em}{}

% Paragraph
\titleformat{\paragraph}[runin]{\normalfont\normalsize\bfseries}{\theparagraph}{1em}{}[.]

% Itemize/enumerate spacing
\RequirePackage{enumitem}
\setlist{noitemsep,topsep=2pt}

% runningheads option (no-op)
\DeclareOption{runningheads}{}
\ProcessOptions\relax
