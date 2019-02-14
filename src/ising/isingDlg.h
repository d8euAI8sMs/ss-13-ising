// isingDlg.h : header file
//

#include <util/common/gui/SimulationDialog.h>
#include <util/common/gui/PlotControl.h>

#include "model.h"

#pragma once

// CIsingDlg dialog
class CIsingDlg : public CSimulationDialog
{
// Construction
public:
    CIsingDlg(CWnd* pParent = NULL);    // standard constructor

// Dialog Data
    enum { IDD = IDD_ISING_DIALOG };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
    HICON m_hIcon;

    // Generated message map functions
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    DECLARE_MESSAGE_MAP()
public:
    CPlotControl m_spinPlot;
    CPlotControl m_energyPlot;
    CPlotControl m_magPlot;
    CPlotControl m_cPlot;
    CPlotControl m_hiPlot;
    model::model_data m_data;
    afx_msg void OnBnClickedButton1();
    afx_msg void OnBnClickedButton2();
    virtual void OnSimulation();
    void OnDemo();
    void OnCalc();
    size_t m_nN;
    size_t m_n;
    double m_J;
    double m_T;
    double m_T2;
    size_t m_S;
    size_t m_M;
    bool m_bFixedT;
    CButton m_bKeyFrames;
    CButton m_bPaint;
    afx_msg void OnBnClickedButton3();
    void CleanPlot();
    void UpdatePlot(double tOrT);
    void UpdateSpins(bool keyframe);
    CEdit m_Te;
    afx_msg void OnBnClickedButton4();
    CButton m_bOpenCLCtrl;
    CButton m_bGpuCtrl;
};
