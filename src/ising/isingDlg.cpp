// isingDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ising.h"
#include "isingDlg.h"
#include "afxdialogex.h"

#include <ctime>
#include <omp.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CIsingDlg dialog

CIsingDlg::CIsingDlg(CWnd* pParent /*=NULL*/)
    : CSimulationDialog(CIsingDlg::IDD, pParent)
    , m_data(model::make_model_data())
    , m_J(m_data.params->J)
    , m_nN(m_data.params->n)
    , m_S(100)
    , m_n(100)
    , m_M(10)
    , m_T(0.5)
    , m_T2(1.5)
    , m_bFixedT(true)
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CIsingDlg::DoDataExchange(CDataExchange* pDX)
{
    CSimulationDialog::DoDataExchange(pDX);
    DDX_Control(pDX, IDC_SPIN, m_spinPlot);
    DDX_Control(pDX, IDC_SPIN2, m_energyPlot);
    DDX_Control(pDX, IDC_SPIN3, m_magPlot);
    DDX_Control(pDX, IDC_SPIN4, m_cPlot);
    DDX_Control(pDX, IDC_SPIN5, m_hiPlot);
    DDX_Control(pDX, IDC_CHECK1, m_bKeyFrames);
    DDX_Control(pDX, IDC_CHECK2, m_bPaint);
    DDX_Control(pDX, IDC_EDIT8, m_Te);
    DDX_Text(pDX, IDC_EDIT1, m_nN);
    DDX_Text(pDX, IDC_EDIT2, m_J);
    DDX_Text(pDX, IDC_EDIT3, m_T);
    DDX_Text(pDX, IDC_EDIT4, m_T2);
    DDX_Text(pDX, IDC_EDIT7, m_n);
    DDX_Text(pDX, IDC_EDIT5, m_S);
    DDX_Text(pDX, IDC_EDIT6, m_M);
}

BEGIN_MESSAGE_MAP(CIsingDlg, CSimulationDialog)
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    ON_BN_CLICKED(IDC_BUTTON1, &CIsingDlg::OnBnClickedButton1)
    ON_BN_CLICKED(IDC_BUTTON2, &CIsingDlg::OnBnClickedButton2)
    ON_BN_CLICKED(IDC_BUTTON3, &CIsingDlg::OnBnClickedButton3)
    ON_BN_CLICKED(IDC_BUTTON4, &CIsingDlg::OnBnClickedButton4)
END_MESSAGE_MAP()

// CIsingDlg message handlers

BOOL CIsingDlg::OnInitDialog()
{
    CSimulationDialog::OnInitDialog();

    // Set the icon for this dialog.  The framework does this automatically
    //  when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);            // Set big icon
    SetIcon(m_hIcon, FALSE);        // Set small icon

    // TODO: Add extra initialization here

    m_data.system_data.init(*m_data.params);

    m_spinPlot.plot_layer.with(model::make_system_plot(m_data.system_data));

    m_energyPlot.plot_layer.with(
        model::make_root_drawable(m_data.e_data, {{ m_data.e_data.plot }})
    );
    m_magPlot.plot_layer.with(
        model::make_root_drawable(m_data.m_data, {{ m_data.m_data.plot }})
    );
    m_cPlot.plot_layer.with(
        model::make_root_drawable(m_data.c_data, {{ m_data.c_data.plot }})
    );
    m_hiPlot.plot_layer.with(
        model::make_root_drawable(m_data.hi_data, {{ m_data.hi_data.plot }})
    );

    m_spinPlot.triple_buffered = true;
    m_energyPlot.triple_buffered = true;
    m_magPlot.triple_buffered = true;
    m_cPlot.triple_buffered = true;
    m_hiPlot.triple_buffered = true;

    m_bPaint.SetCheck(BST_CHECKED);

    return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CIsingDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else
    {
        CSimulationDialog::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CIsingDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}


void CIsingDlg::OnBnClickedButton1()
{
    UpdateData(TRUE);

    m_bFixedT = true;

    StartSimulationThread();
}


void CIsingDlg::OnBnClickedButton2()
{
    StopSimulationThread();
}


void CIsingDlg::OnSimulation()
{
    m_data.params->n = m_nN;
    m_data.params->J = m_J;

    srand(std::time(NULL));

    m_data.system_data.init(*m_data.params);

    CleanPlot();

    if (m_bFixedT)
        OnDemo();
    else
        OnCalc();

    CSimulationDialog::OnSimulation();
}

void CIsingDlg::OnBnClickedButton3()
{
    UpdateData(TRUE);

    m_bFixedT = false;

    StartSimulationThread();
}

void CIsingDlg::CleanPlot()
{
    m_data.e_data.data->clear();
    m_data.m_data.data->clear();
    m_data.c_data.data->clear();
    m_data.hi_data.data->clear();
    m_data.e_data.autoworld->clear();
    m_data.m_data.autoworld->clear();
    m_data.c_data.autoworld->clear();
    m_data.hi_data.autoworld->clear();
}

void CIsingDlg::UpdatePlot(double x)
{
    m_data.e_data.data->emplace_back(x, m_data.system_data.params.e);
    m_data.m_data.data->emplace_back(x, m_data.system_data.params.m);
    m_data.c_data.data->emplace_back(x, m_data.system_data.params.c);
    m_data.hi_data.data->emplace_back(x, m_data.system_data.params.hi);
    m_data.e_data.autoworld->setup(*m_data.e_data.data);
    m_data.m_data.autoworld->setup(*m_data.m_data.data);
    m_data.c_data.autoworld->setup(*m_data.c_data.data);
    m_data.hi_data.autoworld->setup(*m_data.hi_data.data);
    m_energyPlot.RedrawBuffer(); m_energyPlot.SwapBuffers();
    m_magPlot.RedrawBuffer(); m_magPlot.SwapBuffers();
    m_cPlot.RedrawBuffer(); m_cPlot.SwapBuffers();
    m_hiPlot.RedrawBuffer(); m_hiPlot.SwapBuffers();
    Invoke([this] () {
        m_energyPlot.RedrawWindow();
        m_magPlot.RedrawWindow();
        m_cPlot.RedrawWindow();
        m_hiPlot.RedrawWindow();
    });
}

void CIsingDlg::UpdateSpins(bool keyframe)
{
    if (m_bPaint.GetCheck() != BST_CHECKED) return;
    if (keyframe == (m_bKeyFrames.GetCheck() == BST_CHECKED))
    {
        m_spinPlot.RedrawBuffer();
        m_spinPlot.SwapBuffers();
        Invoke([this] () { m_spinPlot.RedrawWindow(); });
    }
}

void CIsingDlg::OnDemo()
{
    size_t time = 0;

    double T = m_data.params->Tc() * m_T;

    m_data.system_data.begin(*m_data.params, T);
    for (size_t i = 0; (i < m_S) && m_bWorking; ++i)
    {
        m_data.system_data.next_opencl();
        UpdateSpins(false);
    }
    m_data.system_data.end();
    UpdateSpins(true);

    while (m_bWorking)
    {
        ++time;
        Invoke([this, &T] () { T = m_data.params->Tc() * m_T; });
        m_data.system_data.begin(*m_data.params, T);
        for (size_t i = 0; (i < m_M) && m_bWorking; ++i)
        {
            m_data.system_data.next_opencl();
            UpdateSpins(false);
        }
        m_data.system_data.end();

        UpdatePlot(time);
        UpdateSpins(true);
    }
}

void CIsingDlg::OnCalc()
{
    const size_t n = m_n;
    const double Tc = m_data.params->Tc();
    const double T0 = m_T;

    double dT = (m_T2 - T0) / n;

    m_data.system_data.begin(*m_data.params, T0);
    for (size_t i = 0; (i < m_S * 10) && m_bWorking; ++i)
    {
        m_data.system_data.next_opencl();
        UpdateSpins(false);
    }
    m_data.system_data.end();
    UpdateSpins(true);

    double maxC = 0, maxT = 0;

    for (size_t t = 0; (t <= n) && m_bWorking; ++t)
    {
        double T = Tc * (dT * t + T0);
        m_data.system_data.begin(*m_data.params, T);
        for (size_t i = 0; (i < m_S) && m_bWorking; ++i)
        {
            m_data.system_data.next_opencl();
            UpdateSpins(false);
        }
        m_data.system_data.end();
        UpdateSpins(true);

        m_data.system_data.begin(*m_data.params, T);
        for (size_t i = 0; (i < m_M) && m_bWorking; ++i)
        {
            m_data.system_data.next_opencl();
            UpdateSpins(false);
        }
        m_data.system_data.end();

        UpdatePlot(T / Tc);
        UpdateSpins(true);

        if (m_data.system_data.params.c > maxC)
        {
            maxC = m_data.system_data.params.c;
            maxT = T / Tc;
            CString fmt; fmt.Format(TEXT("%lf"), maxT);
            m_Te.SetWindowText(fmt);
        }
    }
}


void CIsingDlg::OnBnClickedButton4()
{
    CFileDialog fd(FALSE, TEXT("bmp"), TEXT("board"));
    if (fd.DoModal() == IDOK)
    {
        auto f = fd.GetPathName();
        auto bmp = model::export_system(m_data.system_data, true);
        CImage img;
        img.Attach((HBITMAP) bmp->Detach());
        img.Save(f);
    }
}
