// isingDlg.h : header file
//

#include <util/common/gui/SimulationDialog.h>

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
};
