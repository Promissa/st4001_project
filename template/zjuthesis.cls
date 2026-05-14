\ProvidesClass{zjuthesis}[2025/03/20 v1.0 Zhejiang University Thesis Template]
\newif\ifChecklistFill\ChecklistFillfalse
\DeclareOption{filled}{\ChecklistFilltrue}
% \DeclareOption*{%
%   \PassOptionsToClass{\CurrentOption}{article}
% }
\ProcessOptions\relax

\LoadClass[a4paper,12pt,oneside]{book}





\RequirePackage{geometry}
\geometry{
    a4paper,
    top=2.54cm,
    bottom=2.9cm,
    left=3.17cm,
    top=3.17cm,
}

% set the font
\RequirePackage{fontspec}
\IfFontExistsTF{Times New Roman}
  {\setmainfont{Times New Roman}}
  {\setmainfont{TeX Gyre Termes}}

\RequirePackage{amsmath,amssymb}
% Equation numbering style: (Chapter-Equation), e.g., (2-1), (2-2)
\numberwithin{equation}{chapter}
\renewcommand{\theequation}{\thechapter-\arabic{equation}}
\RequirePackage{graphicx}
\RequirePackage{ulem}
\RequirePackage{setspace}
\RequirePackage{indentfirst}

\DeclareRobustCommand{\chuhao}{\fontsize{42pt}{50.4pt}\selectfont}
\DeclareRobustCommand{\xiaochu}{\fontsize{36pt}{43.2pt}\selectfont}
\DeclareRobustCommand{\yihao}{\fontsize{26pt}{31.2pt}\selectfont}
\DeclareRobustCommand{\xiaoyi}{\fontsize{24pt}{28.8pt}\selectfont}
\DeclareRobustCommand{\erhao}{\fontsize{22pt}{26.4pt}\selectfont}
\DeclareRobustCommand{\xiaoer}{\fontsize{18pt}{21.6pt}\selectfont}
\DeclareRobustCommand{\sanhao}{\fontsize{16pt}{19.2pt}\selectfont}
\DeclareRobustCommand{\xiaosan}{\fontsize{15pt}{18pt}\selectfont}
\DeclareRobustCommand{\sihao}{\fontsize{14pt}{16.8pt}\selectfont}
\DeclareRobustCommand{\shisanbang}{\fontsize{13pt}{15.pt}\selectfont}
\DeclareRobustCommand{\xiaosi}{\fontsize{12pt}{14.4pt}\selectfont}
\DeclareRobustCommand{\wuhao}{\fontsize{10.5pt}{12.6pt}\selectfont}
\DeclareRobustCommand{\xiaowu}{\fontsize{9pt}{10.8pt}\selectfont}
\DeclareRobustCommand{\liuhao}{\fontsize{7.5pt}{9pt}\selectfont}
\DeclareRobustCommand{\xiaoliu}{\fontsize{6.5pt}{7.8pt}\selectfont}
\DeclareRobustCommand{\qihao}{\fontsize{5.5pt}{6.6pt}\selectfont}
\DeclareRobustCommand{\bahao}{\fontsize{5pt}{6pt}\selectfont}

\RequirePackage{keyval}

\def\zju@title{}
\def\zju@name{}
\def\zju@id{}
\def\zju@supervisor{}
\def\zju@classOfYear{}
\def\zju@major{}
\def\zju@submissionDate{}
\def\zju@academicTitle{}

\define@key{zju}{title}{\def\zju@title{#1}}
\define@key{zju}{name}{\def\zju@name{#1}}
\define@key{zju}{id}{\def\zju@id{#1}}
\define@key{zju}{supervisor}{\def\zju@supervisor{#1}}
\define@key{zju}{class of year}{\def\zju@classOfYear{#1}}
\define@key{zju}{major}{\def\zju@major{#1}}
\define@key{zju}{submission date}{\def\zju@submissionDate{#1}}
\define@key{zju}{Academic title}{\def\zju@academicTitle{#1}}

\newcommand\zju[1]{%
    \setkeys{zju}{#1}%
}

% 封面
\def\maketitle{
    \begin{titlepage}
        {\hfill \fontsize{11pt}{17.16pt}\selectfont Confidential\, thesis\, $\square$ }

        \vspace{1cm}
        \centerline{\includegraphics[width=10.8cm]{images/logo.png}}
        \centerline{\sihao Zhejiang University-University of Illinois Urbana-Champaign Institute}

        \vspace{1.5cm}
        \centerline{\xiaochu Undergraduate Thesis }\vspace{10pt}
        \centerline{\xiaochu (Design)}

        \vspace{2cm}
        \centerline{\includegraphics[width=4.61cm]{images/logo1.png}}

        \vspace{1.2cm}
        \begin{center}
            \sihao
            \renewcommand{\arraystretch}{1.75}
            \setlength{\tabcolsep}{0pt}
            \begin{tabular}{p{3.5cm}p{10.89cm}}
                Thesis Title    & \uline{\hfill \zju@title \hfill}          \\
                Student Name    & \uline{\hfill \zju@name \hfill}           \\
                Student ID      & \uline{\hfill \zju@id \hfill}             \\
                Supervisor      & \uline{\hfill \zju@supervisor \hfill}     \\
                Class of year   & \uline{\hfill \zju@classOfYear \hfill}    \\
                Major           & \uline{\hfill \zju@major \hfill}          \\
                Submission date & \uline{\hfill \zju@submissionDate \hfill} \\
            \end{tabular}
        \end{center}

    \end{titlepage}
}

% 页眉
\RequirePackage{fancyhdr}
\fancypagestyle{ZJUThesis}{
    % 页眉高度
    \setlength{\voffset}{-10mm}
    \setlength{\headheight}{14pt}
    \setlength{\topmargin}{0mm}
    \setlength{\headsep}{8mm}
    \setlength{\footskip}{12pt}


    \fancyhf{}
    % 定义页眉、页码
    \fancyhead[L]{\xiaowu Class of 2026}
    \fancyhead[C]{\xiaowu \zju@name}
    \fancyhead[R]{\xiaowu \zju@id}
    \fancyfoot[C]{\xiaowu \thepage}
    % 页眉分割线粗细
    \renewcommand{\headrulewidth}{0.7pt}

}


\renewcommand{\frontmatter}{
    \pagenumbering{Roman}
    \pagestyle{ZJUThesis}
}

\RequirePackage{multirow,makecell}


\newcommand{\Checklist}{
    \bgroup
    \clearpage

    \centerline{\xiaoer\bf Content Checklist}
    \vspace{1em}

    \noindent (Student should bind this form on the first page of the thesis after the cover page. The form should be filled out by the supervisor after the oral defense.)

    \fontsize{11pt}{13.2pt}\selectfont
    \renewcommand{\arraystretch}{1.75}
    \vspace{2em}
    \noindent
    \begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}
        \hline
        \multicolumn{2}{|l|}{Student Name}                                                  & \makebox[2.5cm]{\ifChecklistFill\zju@name\else \fi}                          & ZJU ID & \multicolumn{3}{c|}{\makebox[2.5cm]{\ifChecklistFill\zju@id\else\fi}} & \multicolumn{3}{c|}{Major} & \multicolumn{2}{c|}{\makebox[2.5cm]{\ifChecklistFill\zju@major\else\fi}} \\
        \hline
        \multicolumn{2}{|l|}{Supervisor}                                                    & \multicolumn{4}{c|}{\makebox[4cm]{\makecell{~                                                                                                                                                                                                                         \\\ifChecklistFill\zju@supervisor\else\fi\\~\\}}}                                         & \multicolumn{2}{c|}{\makecell[c]{Academic                                                                                                                \\ Title}} & \multicolumn{4}{c|}{\makebox[3cm]{\ifChecklistFill\zju@academicTitle\else\fi}}                                                                                    \\
        \hline
        \multicolumn{2}{|l|}{Thesis title}                                                  & \multicolumn{10}{c|}{\makecell{~                                                                                                                                                                                                                                      \\\ifChecklistFill\zju@title\else\fi\\~\\}}                                                                                                                                                                                                           \\
        \hline
        \multicolumn{10}{|c|}{\xiaosi\textbf{Content Checklist}}                            & \multicolumn{2}{c|}{\xiaosi\bf\makecell{Evaluation                                                                                                                                                                                                                    \\ Section (√)}}                                                                                                                                                        \\
        \hline
        \multirow{14}{*}{\xiaosi\bf\makebox[1cm]{\rotatebox[origin=c]{-90}{Senior Thesis}}} & \multicolumn{9}{l|}{1. Cover (Use the unified cover, blue recommended)}      & {Good} & Fair                                                                                                                                                                          \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{2. Academic integrity Commitment of Senior Thesis}       & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{3.    Acknowledgements}                                  & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{4. Abstract}                                             & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{5. Table of Contents}                                    & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{6. Main Content (Page numbers start from the main text)} & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{7. Reference}                                            & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{8. Appendix (if need)}                                   & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{9.   Author's Biography}                                 & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{10. Task plan of senior thesis}                          & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{11. Assessment Form of senior thesis}                    & {}     &                                                                                                                                                                               \\
        \cline{2-12}                                                                        & \multicolumn{9}{l|}{12. On-site Minutes Form of Oral Defense}                & {}     &                                                                                                                                                                               \\
        \hline
        \multicolumn{6}{|l|}{Signature of Supervisor: \makebox[4.5cm]{\makecell[t]{~                                                                                                                                                                                                                                                                                \\~\\~\\~\\}}}                           & \multicolumn{6}{l|}{Date:} \\
        \hline
    \end{tabular}
    \egroup
}

\newcommand{\Commitment}{
    \bgroup
    \clearpage
    \centerline{\includegraphics[width=9.85cm]{images/logo2.png}}
    \vspace{1.2cm}
    \centerline{\bfseries\xiaoer Academic integrity Commitment of Senior Thesis }
    \vspace{1.2cm}

    \setlength{\parskip}{10pt}
    \xiaosi

    \noindent 1. I solemnly promise that the submitted thesis has been completed under the guidance of my supervisor in strict accordance with the relevant regulations of the university and the college.

    \noindent 2. In my thesis, except for the parts specifically marked and acknowledged, it does not contain research results that have been published or written by others, nor does it contain materials used to obtain degrees or certificates from Zhejiang University or other educational institutions.

    \noindent 3. Any contributions made by my colleagues to this research have been clearly stated and acknowledged in the thesis.

    \noindent 4. I promise that there has been no fabrication of data in the process of completing the thesis.

    \noindent 5. If there is any infringement of intellectual property rights in this thesis, I shall bear the corresponding legal responsibilities.

    \noindent 6. I fully understand that Zhejiang University has the right to retain and submit copies and disks of this thesis (design) to relevant departments or agencies, and to allow the thesis (design) to be consulted and borrowed. I authorize Zhejiang University to include all or part of this thesis (design) in relevant databases for retrieval and dissemination, and to save and compile this thesis (design) by means of photocopying, reduction, or scanning.

    \vspace{2.2cm}
    \noindent
    \begin{tabular}{lp{3.5cm}lp{5cm}}
        Author's Signature: &  & Supervisor's Signature: & \\
        \\
        \\
        Date:               &  & Date:                   & \\
    \end{tabular}
    \egroup
}


\renewcommand{\mainmatter}{
    \pagenumbering{arabic}
    \pagestyle{ZJUThesis}
    \doublespacing
}

\RequirePackage{titlesec}
\fancypagestyle{plain}{
    \pagestyle{fancy}
}

\titlespacing*{\chapter}{0pt}{-30pt}{0pt}% \titlespacing*{章节命令}{左边距}{上文距}{下文距}[右边距]
\titleformat{\chapter}
{\bfseries\xiaoer}
{Chapter \thechapter. }
{0pt}
{}

\titlespacing*{\section}{0pt}{0pt}{5pt}
\titleformat{\section}
{\bfseries\sihao}
{\thesection\;}
{0pt}
{}

\titlespacing*{\subsection}{0pt}{0pt}{5pt}
\titleformat{\subsection}
{\xiaosi}
{\thesubsection\;}
{0pt}
{}


\RequirePackage{setspace}
% \doublespacing

\RequirePackage{caption}
\captionsetup[figure]{labelfont={bf,normalsize},labelsep=space,justification=raggedright,singlelinecheck=true}
\captionsetup[table]{labelfont={bf,normalsize},labelsep=space,justification=raggedright,singlelinecheck=true}

\RequirePackage{array}

\renewcommand{\contentsname}{\xiaoyi\bf\vspace{0.8em} Content}
\RequirePackage{titletoc}
% \titlecontens{chapter}[<left>]{<above>}{⟨before with label⟩}{⟨before without label⟩}{⟨filler and page⟩}[⟨after⟩]
\titlecontents{chapter}[0em]{\xiaosi}{Chapter \thecontentslabel.\;}{}{\hspace{.5em}\titlerule*{.}\contentspage}[\vspace{5pt}]
\titlecontents{section}[1em]{\xiaosi}{\thecontentslabel\;}{}{\hspace{.5em}\titlerule*{.}\contentspage}[\vspace{5pt}]
\titlecontents{subsection}[2em]{\xiaosi}{\thecontentslabel\;}{}{\hspace{.5em}\titlerule*{.}\contentspage}[\vspace{5pt}]
\setcounter{tocdepth}{2}

\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{hyperref}
\urlstyle{same}

% \renewcommand{cmd}{def}

\newdimen\bibspacing
\setlength\bibspacing\z@
\renewenvironment{thebibliography}[1]{%
 \bibfont\bibsection\parindent \z@\list
   {\@biblabel{\arabic{NAT@ctr}}}{\@bibsetup{#1}%
    \setcounter{NAT@ctr}{0}}%
    \ifNAT@openbib
      \renewcommand\newblock{\par}
    \else
      \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
    \fi
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax
    \let\citeN\cite \let\shortcite\cite
    \let\citeasnoun\cite
            \itemsep\bibspacing %
            \parsep\z@skip %
 }{\def\@noitemerr{%
  \PackageWarning{natbib}
     {Empty `thebibliography' environment}}%
  \endlist\vskip-\lastskip}