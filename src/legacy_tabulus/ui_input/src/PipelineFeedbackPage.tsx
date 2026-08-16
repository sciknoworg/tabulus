import React, { useState } from 'react';
import styles from './PipelineFeedbackPage.module.css';
import Papa from 'papaparse';

type StepStatus = 'idle' | 'pending' | 'running' | 'success' | 'error';

type TableItem = {
  table_id?: number;
  page_nr?: number | null;
  in_references?: boolean;
  png?: string;
  png_name?: string;
  bbox?: unknown;
  table_caption?: string | null;
  table_footnote?: string | null;
};

type OcrTable = {
  table_id?: number | null;
  source_file?: string;
  n_rows?: number;
  n_cols?: number;
  rows?: string[][];
  source?: string;
  raw?: string;
  error?: string;
  is_reference_table?: boolean;
  has_tag_match?: boolean;
  has_citation_match?: boolean;
  matched_header_cells?: string[];
  matched_citation_cells?: string[];
  reason?: string;
};

type UploadResponse = {
  job_id: number;
  status: string;
  original_name: string;
  stored_pdf_path: string;
  processing_dir: string;
  refs_start_page: number | null;
  tables_found: number;
  crops_saved: number;
  tables: TableItem[];
  ocr_tables_found?: number;
  ocr_tables?: OcrTable[];
  reference_tables_found?: number;
};

type ResolvedCsv = {
  csv_name: string;
  replacements: number;
};

type CsvPreview = {
  csv_name: string;
  rows: string[][];
};

type ReferenceMatch = {
  row_index?: number;
  value?: string;
  table_reference?: string;

  found: boolean;

  matched_reference_indices?: number[];
  matched_references?: string[];

  doi?: string[];

  is_header?: boolean;

  kreuzberg_used?: boolean;
  kreuzberg_pattern_used?: string;
};

type ReferenceMatchResponse = {
  job_id: number;
  status: string;

  bibliography_count: number;

  bibliography_source?: string;

  kreuzberg_used?: boolean;
  kreuzberg_reason?: string;
  kreuzberg_pattern_used?: string;

  reference_tables_checked?: number;

  matched_tables: {
    source_file?: string;
    table_id?: number;
    reference_column_index?: number;
    matches_found?: number;
    matches_total?: number;
    matches?: ReferenceMatch[];
  }[];

  resolved_csvs?: ResolvedCsv[];
};


const BACKEND_URL = 'http://localhost:8000';

function StatusBadge({ status }: { status: StepStatus }) {
  return <span className={styles.statusBadge}>{status}</span>;
}

function StepCard({
  title,
  status,
  message,
  children,
}: {
  title: string;
  status: StepStatus;
  message?: string;
  children?: React.ReactNode;
}) {

  return (
    <div className={styles.stepCard}>
      <div className={styles.stepHeader}>
        <span className={`${styles.stepDot} ${styles[`dot_${status}`]}`} />
        <h3 className={styles.stepTitle}>{title}</h3>
        <StatusBadge status={status} />
      </div>

      {message && <p className={styles.stepMessage}>{message}</p>}
      {children}
    </div>
  );
}

function ImageModal({
  isOpen,
  imageUrl,
  imageAlt,
  onClose,
}: {
  isOpen: boolean;
  imageUrl: string | null;
  imageAlt: string;
  onClose: () => void;
}) {
  if (!isOpen || !imageUrl) return null;

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <button className={styles.modalCloseButton} onClick={onClose}>
          ×
        </button>
        <img src={imageUrl} alt={imageAlt} className={styles.modalImage} />
      </div>
    </div>
  );
}

function OcrTablePreview({ ocrMatch }: { ocrMatch: OcrTable | null }) {
  if (ocrMatch?.error) {
    return <div className={styles.errorBox}>{ocrMatch.error}</div>;
  }

  if (ocrMatch?.rows && ocrMatch.rows.length > 0) {
    return (
      <div className={styles.tableWrapper}>
        <table className={styles.dataTable}>
          <tbody>
            {ocrMatch.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className={`${styles.dataCell} ${
                      rowIndex === 0 ? styles.headerCell : ''
                    }`}
                  >
                    {cell || '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (ocrMatch?.raw) {
    return <pre className={styles.rawBox}>{ocrMatch.raw}</pre>;
  }

  return <div className={styles.emptyBox}>No OCR table extracted</div>;
}

export default function PipelineFeedbackPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [uploadStatus, setUploadStatus] = useState<StepStatus>('idle');
  const [uploadMessage, setUploadMessage] = useState('Please choose a PDF file.');

  const [mineruVisible, setMineruVisible] = useState(false);
  const [mineruStatus, setMineruStatus] = useState<StepStatus>('idle');
  const [mineruMessage, setMineruMessage] = useState('Waiting for PDF upload.');

  const [ocrVisible, setOcrVisible] = useState(false);
  const [ocrStatus, setOcrStatus] = useState<StepStatus>('idle');
  const [ocrMessage, setOcrMessage] = useState('Waiting for MinerU crop.');

  const [refVisible, setRefVisible] = useState(false);
  const [refStatus, setRefStatus] = useState<StepStatus>('idle');
  const [refMessage, setRefMessage] = useState('Waiting for PaddleOCR output.');

  const [matchVisible, setMatchVisible] = useState(false);
  const [matchStatus, setMatchStatus] = useState<StepStatus>('idle');
  const [matchMessage, setMatchMessage] = useState('Waiting for reference-table detection.');
  const [matchResult, setMatchResult] = useState<ReferenceMatchResponse | null>(null);

  const [result, setResult] = useState<UploadResponse | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [modalImageUrl, setModalImageUrl] = useState<string | null>(null);
  const [modalImageAlt, setModalImageAlt] = useState('');
  const [csvPreviews, setCsvPreviews] = useState<CsvPreview[]>([]);


const parseCsv = (text: string): string[][] => {
  const parsed = Papa.parse<string[]>(text, {
    skipEmptyLines: true,
  });

  return parsed.data;
};
  const resetStateForNewFile = () => {
    setResult(null);
    setCsvPreviews([]);
    setMineruVisible(false);
    setMineruStatus('idle');
    setMineruMessage('Waiting for PDF upload.');

    setOcrVisible(false);
    setOcrStatus('idle');
    setOcrMessage('Waiting for MinerU crop.');

    setRefVisible(false);
    setRefStatus('idle');
    setRefMessage('Waiting for PaddleOCR output.');

    setMatchVisible(false);
    setMatchStatus('idle');
    setMatchMessage('Waiting for reference-table detection.');
    setMatchResult(null);

    setModalOpen(false);
    setModalImageUrl(null);
    setModalImageAlt('');
  };

  const loadCsvPreviews = async (jobId: number, csvs: ResolvedCsv[]) => {
  const previews = await Promise.all(
    csvs.map(async (csv) => {
      const response = await fetch(
        `${BACKEND_URL}/jobs/${jobId}/resolved-csv/${encodeURIComponent(csv.csv_name)}`
      );

      const text = await response.text();

      return {
        csv_name: csv.csv_name,
        rows: parseCsv(text),
      };
    })
  );

  setCsvPreviews(previews);
};
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    resetStateForNewFile();

    if (!file) {
      setUploadStatus('idle');
      setUploadMessage('Please choose a PDF file.');
      return;
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('error');
      setUploadMessage('Only PDF files are allowed.');
      return;
    }

    setUploadStatus('pending');
    setUploadMessage(`Selected file: ${file.name}`);
  };

  const parseJsonResponse = async (response: Response) => {
    const text = await response.text();
    let data: unknown;

    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Backend did not return JSON. Response: ${text}`);
    }

    if (!response.ok) {
      const detail =
        typeof data === 'object' && data !== null && 'detail' in data
          ? String((data as { detail?: unknown }).detail)
          : 'Request failed.';
      throw new Error(detail);
    }

    return data;
  };

  const runPaddleAutomatically = async (jobId: number) => {
    setOcrVisible(true);
    setOcrStatus('running');
    setOcrMessage('Sending cropped table images to PaddleOCR...');

    setRefVisible(true);
    setRefStatus('running');
    setRefMessage('Checking OCR tables for reference-like content...');

    const response = await fetch(`${BACKEND_URL}/jobs/${jobId}/run-paddle`, {
      method: 'POST',
    });

    const data = (await parseJsonResponse(response)) as UploadResponse;
    setResult(data);

    setOcrStatus('success');
    setOcrMessage(
      `PaddleOCR finished. Extracted ${data.ocr_tables_found ?? 0} OCR table result(s).`
    );

    setRefStatus('success');
    setRefMessage(
      `Reference-table check finished. Found ${
        data.reference_tables_found ?? 0
      } reference-like table(s).`
    );

    return data;
  };

  const runReferenceMatchingAutomatically = async (jobId: number) => {
    setMatchVisible(true);
    setMatchStatus('running');
    setMatchMessage('Sending PDF to GROBID and matching table references...');

    const response = await fetch(
      `${BACKEND_URL}/jobs/${jobId}/match-references?use_crossref=true`,
      { method: 'POST' }
    );

    const data = (await parseJsonResponse(response)) as ReferenceMatchResponse;
    setMatchResult(data);
    if (data.resolved_csvs?.length) {
      await loadCsvPreviews(data.job_id, data.resolved_csvs);
    }

    setMatchStatus('success');
    setMatchMessage(
      `GROBID matching finished. Bibliography references: ${
        data.bibliography_count
      }. CSV files created: ${data.resolved_csvs?.length ?? 0}.`
    );

    return data;
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('error');
      setUploadMessage('Please select a PDF first.');
      return;
    }

    if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('error');
      setUploadMessage('Only PDF files are allowed.');
      return;
    }

    try {
      setUploadStatus('running');
      setUploadMessage('Uploading PDF to backend...');

      setMineruVisible(true);
      setMineruStatus('running');
      setMineruMessage('Running MinerU crop...');

      setOcrVisible(true);
      setOcrStatus('idle');
      setOcrMessage('Waiting for MinerU crop...');

      setRefVisible(true);
      setRefStatus('idle');
      setRefMessage('Waiting for PaddleOCR output...');

      setMatchVisible(true);
      setMatchStatus('idle');
      setMatchMessage('Waiting for reference-table detection.');

      const formData = new FormData();
      formData.append('file', selectedFile);

      const uploadResponse = await fetch(`${BACKEND_URL}/upload-pdf`, {
        method: 'POST',
        body: formData,
      });

      const uploadData = (await parseJsonResponse(uploadResponse)) as UploadResponse;
      setResult(uploadData);

      setUploadStatus('success');
      setUploadMessage(`PDF uploaded successfully: ${uploadData.original_name}`);

      setMineruStatus('success');
      setMineruMessage(
        `MinerU crop finished. refs_start_page: ${
          uploadData.refs_start_page ?? 'not found'
        }, tables_found: ${uploadData.tables_found}`
      );

      await runPaddleAutomatically(uploadData.job_id);
      await runReferenceMatchingAutomatically(uploadData.job_id);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';

      if (uploadStatus === 'running') {
        setUploadStatus('error');
        setUploadMessage(message);
      }

      if (mineruStatus === 'running') {
        setMineruStatus('error');
        setMineruMessage(message);
      }

      if (ocrStatus === 'running') {
        setOcrStatus('error');
        setOcrMessage(message);
      }

      if (refStatus === 'running') {
        setRefStatus('error');
        setRefMessage(message);
      }

      if (matchStatus === 'running') {
        setMatchStatus('error');
        setMatchMessage(message);
      }
    }
  };

  const openImageModal = (imageUrl: string, imageAlt: string) => {
    setModalImageUrl(imageUrl);
    setModalImageAlt(imageAlt);
    setModalOpen(true);
  };

  const closeImageModal = () => {
    setModalOpen(false);
    setModalImageUrl(null);
    setModalImageAlt('');
  };

  const hasOcrResults = (result?.ocr_tables?.length ?? 0) > 0;

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <h1 className={styles.pageTitle}>Pipeline UI</h1>
        <p className={styles.pageSubtitle}>
          Upload a PDF, crop tables with MinerU, extract rows with PaddleOCR,
          detect reference-like tables, match them with GROBID references, and
          create resolved CSV files.
        </p>

        <StepCard title="PDF Upload" status={uploadStatus} message={uploadMessage}>
          <div className={styles.uploadArea}>
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={handleFileChange}
              className={styles.fileInput}
            />

            <button
              onClick={handleUpload}
              className={styles.primaryButton}
              disabled={uploadStatus === 'running' || ocrStatus === 'running' || matchStatus === 'running'}
            >
              Run Full Pipeline
            </button>
          </div>
        </StepCard>

        {mineruVisible && (
          <StepCard title="MinerU Crop" status={mineruStatus} message={mineruMessage}>
            {result && (
              <div className={styles.summaryCard}>
                <p><strong>Job ID:</strong> {result.job_id}</p>
                <p><strong>Refs start page:</strong> {result.refs_start_page ?? 'not found'}</p>
                <p><strong>Tables found:</strong> {result.tables_found}</p>
                <p><strong>Crops saved:</strong> {result.crops_saved}</p>
              </div>
            )}
          </StepCard>
        )}

        {ocrVisible && (
          <StepCard title="PaddleOCR Table Extraction" status={ocrStatus} message={ocrMessage}>
            {result && (
              <div className={styles.summaryCard}>
                <p><strong>OCR tables found:</strong> {result.ocr_tables_found ?? 0}</p>
              </div>
            )}
          </StepCard>
        )}

        {refVisible && (
          <StepCard title="Reference Table Check" status={refStatus} message={refMessage}>
            {result && (
              <div className={styles.summaryCard}>
                <p>
                  <strong>Reference-like tables found:</strong>{' '}
                  {result.reference_tables_found ?? 0}
                </p>
              </div>
            )}
          </StepCard>
        )}

        {matchVisible && (
          <StepCard title="GROBID Reference Matching" status={matchStatus} message={matchMessage}>
            {matchResult && (
              <div className={styles.summaryCard}>
                <p><strong>Bibliography references:</strong> {matchResult.bibliography_count}</p>
                <p>
                  <strong>Reference tables checked:</strong>{' '}
                  {matchResult.reference_tables_checked ?? matchResult.matched_tables.length}
                </p>
                <p><strong>Resolved CSV files:</strong> {matchResult.resolved_csvs?.length ?? 0}</p>
              </div>
            )}
           {matchResult?.kreuzberg_used && (
            <div className={styles.kreuzbergInfoBox}>
              <strong>Kreuzberg fallback used</strong>

              <p>{matchResult.kreuzberg_reason}</p>

              <p>
                <strong>Detected bibliography pattern:</strong>{' '}
                {matchResult.kreuzberg_pattern_used ?? 'unknown'}
              </p>
            </div>
          )}
          </StepCard>
        )}

        {result && result.tables.length > 0 && (
          <>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Cropped table images</h2>
              <p className={styles.sectionText}>Click an image to open it larger.</p>
            </div>

            <div className={styles.cardGrid}>
              {result.tables.map((table, index) => {
                const imageName = table.png_name;
                const imageUrl = imageName
                  ? `${BACKEND_URL}/jobs/${result.job_id}/images/${encodeURIComponent(imageName)}`
                  : null;

                return (
                  <div key={imageName ?? index} className={styles.resultCard}>
                    <div className={styles.metaBlock}>
                      <p><strong>Table:</strong> {table.table_id ?? index + 1}</p>
                      <p><strong>Page:</strong> {table.page_nr ?? '-'}</p>
                      {/* <p><strong>In references:</strong> {table.in_references ? 'yes' : 'no'}</p>*/}
                      <p className={styles.fileName}>{table.png_name ?? 'no image name'}</p>
                    </div>

                    {imageUrl ? (
                      <button
                        type="button"
                        className={styles.imageButton}
                        onClick={() =>
                          openImageModal(imageUrl, table.png_name ?? `table-${index + 1}`)
                        }
                      >
                        <img
                          src={imageUrl}
                          alt={table.png_name ?? `table-${index + 1}`}
                          className={styles.previewImage}
                        />
                      </button>
                    ) : (
                      <div className={styles.emptyBox}>No preview available</div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}

        {result && result.tables.length > 0 && (
          <>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Extracted table text</h2>
              <p className={styles.sectionText}>
                Each OCR result is shown separately with the reference-table decision.
              </p>
            </div>

            <div className={styles.ocrGrid}>
              {result.tables.map((table, index) => {
                const ocrMatch =
                  result.ocr_tables?.find((ocr) => ocr.source_file === table.png_name) ?? null;

                return (
                  <div key={`ocr-${table.png_name ?? index}`} className={styles.ocrCard}>
                    <div className={styles.ocrCardHeader}>
                      <div>
                        <h3 className={styles.ocrCardTitle}>
                          Table {table.table_id ?? index + 1}
                        </h3>
                        <span className={styles.ocrCardMeta}>
                          Page {table.page_nr ?? '-'}
                        </span>
                      </div>

                      <span
                        className={
                          ocrMatch?.is_reference_table
                            ? styles.refBadgeYes
                            : styles.refBadgeNo
                        }
                      >
                        {ocrMatch?.is_reference_table ? 'reference-like' : 'not reference-like'}
                      </span>
                    </div>

                    <p className={styles.fileName}>{table.png_name ?? 'no image name'}</p>

                    {ocrMatch && (
                      <div className={styles.detectionBox}>
                        <p><strong>Decision:</strong> {ocrMatch.reason ?? '-'}</p>
                        <p>
                          <strong>Header evidence:</strong>{' '}
                          {ocrMatch.matched_header_cells?.length
                            ? ocrMatch.matched_header_cells.join(' | ')
                            : 'none'}
                        </p>
                        <p>
                          <strong>Citation evidence:</strong>{' '}
                          {ocrMatch.matched_citation_cells?.length
                            ? ocrMatch.matched_citation_cells.join(' | ')
                            : 'none'}
                        </p>
                      </div>
                    )}

                    <OcrTablePreview ocrMatch={ocrMatch} />
                  </div>
                );
              })}
            </div>
          </>
        )}

       {matchResult?.matched_tables && matchResult.matched_tables.length > 0 && (
          <>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>GROBID reference matches</h2>
              <p className={styles.sectionText}>
                Reference-column values matched against the bibliography extracted by GROBID.
              </p>
            </div>

            <div className={styles.ocrGrid}>
              {matchResult.matched_tables.map((table, index) => (
                <div key={`${table.source_file ?? index}-match`} className={styles.ocrCard}>
                  <h3 className={styles.ocrCardTitle}>
                    {table.source_file ?? `Matched table ${index + 1}`}
                  </h3>

                  <p><strong>Reference column:</strong> {table.reference_column_index ?? '-'}</p>
                  <p>
                    <strong>Matches found:</strong>{' '}
                    {table.matches_found ?? table.matches?.filter((m) => m.found).length ?? 0}
                    {' / '}
                    {table.matches_total ?? table.matches?.length ?? 0}
                  </p>

                  {table.matches && table.matches.length > 0 && (
                    <div className={styles.tableWrapper}>
                      <table className={styles.dataTable}>
                        <thead>
                          <tr>
                            <th className={styles.dataCell}>Table ref</th>
                            <th className={styles.dataCell}>Matched bibliography ref</th>
                            <th className={styles.dataCell}>DOI / URL</th>
                            <th className={styles.dataCell}>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {table.matches.map((match, matchIndex) => (
                              <tr key={matchIndex}>
                                <td className={styles.dataCell}>
                                  {match.value ?? match.table_reference ?? '-'}
                                </td>

                                <td className={styles.dataCell}>
                                  {match.matched_references?.length ? (
                                      <div className={styles.multiMatchList}>
                                        {match.matched_references.map((ref, idx) => (
                                            <div key={idx} className={styles.multiMatchItem}>
                                              [{match.matched_reference_indices?.[idx] ?? '?'}] {ref}
                                            </div>
                                        ))}
                                      </div>
                                  ) : (
                                      '-'
                                  )}
                                </td>

                                <td className={styles.dataCell}>
                                  {match.doi?.length ? (
                                      <div className={styles.multiMatchList}>
                                        {match.doi.map((doi, idx) => (
                                            <div key={idx} className={styles.multiMatchItem}>
                                              {doi.startsWith('http') ? (
                                                  <a href={doi} target="_blank" rel="noreferrer">
                                                    {doi}
                                                  </a>
                                              ) : (
                                                  doi
                                              )}
                                            </div>
                                        ))}
                                      </div>
                                  ) : (
                                      '-'
                                  )}
                                </td>

                                <td className={styles.dataCell}>
                                  {match.found ? 'found' : 'not found'}
                                </td>
                              </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
       )}
        {matchResult?.resolved_csvs && matchResult.resolved_csvs.length > 0 && (
            <>
              <div className={styles.sectionHeader}>
                <h2 className={styles.sectionTitle}>Resolved CSV files</h2>
                <p className={styles.sectionText}>
                  The reference column was replaced with DOI values where matches were found.
                </p>
              </div>

              <div className={styles.ocrGrid}>
                {matchResult.resolved_csvs.map((csv) => (
                    <div key={csv.csv_name} className={styles.ocrCard}>
                      <h3 className={styles.ocrCardTitle}>{csv.csv_name}</h3>
                      <p>
                        <strong>DOI replacements:</strong> {csv.replacements}
                      </p>

                      <a
                          className={styles.downloadButton}
                          href={`${BACKEND_URL}/jobs/${matchResult.job_id}/resolved-csv/${encodeURIComponent(
                        csv.csv_name
                      )}`}
                      download={csv.csv_name}
                    >
                      Download CSV
                    </a>
                    {csvPreviews.find((preview) => preview.csv_name === csv.csv_name)
  ?.rows?.length ? (
  <div className={styles.tableWrapper}>
    <table className={styles.dataTable}>
      <tbody>
        {csvPreviews
          .find((preview) => preview.csv_name === csv.csv_name)!
          .rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className={`${styles.dataCell} ${
                    rowIndex === 0 ? styles.headerCell : ''
                  }`}
                >
                  {cell || '-'}
                </td>
              ))}
            </tr>
          ))}
      </tbody>
    </table>
  </div>
) : null}
                  </div>
                ))}
              </div>
            </>
          )}
        {result && result.tables.length === 0 && mineruStatus === 'success' && (
          <div className={styles.infoBox}>
            MinerU finished, but no cropped table images were found.
          </div>
        )}

        {result && result.tables.length > 0 && !hasOcrResults && ocrStatus === 'success' && (
          <div className={styles.infoBox}>
            PaddleOCR finished, but no parseable table rows were found.
          </div>
        )}
      </div>

      <ImageModal
        isOpen={modalOpen}
        imageUrl={modalImageUrl}
        imageAlt={modalImageAlt}
        onClose={closeImageModal}
      />
    </div>
  );
}