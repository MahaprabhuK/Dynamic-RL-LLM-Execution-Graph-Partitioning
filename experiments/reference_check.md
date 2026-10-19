# Reference check for docs/report/final_report.html (19 October 2026)

Rule: cite only papers that were opened, and check title, authors, venue, year and DOI/arXiv on the
publisher or arXiv page. IEEE Xplore answered automated requests with HTTP 202 (bot challenge) and empty
pages, ACM DL with HTTP 403, and dblp with an access-denied page, so for IEEE papers the paper itself was
opened from an author copy, and venue, pages and DOI were taken from the publisher-deposited Crossref DOI
record. That is noted per entry.

## Cited

| # | Reference | Opened | Checked against | Status |
|---|---|---|---|---|
| 1 | P. Zhang, G. Zeng, T. Wang, W. Lu. *TinyLlama: An Open-Source Small Language Model.* arXiv:2401.02385, 2024 (technical report). | arXiv abstract page | https://arxiv.org/abs/2401.02385 : title, 4 authors, v1 4 Jan 2024, no venue (comments: "Technical Report") | verified |
| 2 | V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, D. Hassabis. *Human-level control through deep reinforcement learning.* Nature 518(7540):529-533, 2015. doi:10.1038/nature14236 | Nature article page (curl, citation meta tags) | https://www.nature.com/articles/nature14236 : title, journal Nature, volume 518, first page 529, date 2015/02, DOI. Author list, issue 7540 and last page 533 from https://api.openalex.org/works/doi:10.1038/nature14236 | verified (authors/issue/last page via OpenAlex) |
| 3 | L. Li, W. Chu, J. Langford, R. E. Schapire. *A Contextual-Bandit Approach to Personalized News Article Recommendation.* Proc. 19th International Conference on World Wide Web (WWW 2010), pp. 661-670. doi:10.1145/1772690.1772758; arXiv:1003.0146 | arXiv abstract page | https://arxiv.org/abs/1003.0146 : title, 4 authors, 2010, journal-ref WWW 2010, related DOI 10.1145/1772690.1772758. Pages 661-670 from Crossref (https://api.crossref.org/works?query.bibliographic=...). ACM page https://dl.acm.org/doi/10.1145/1772690.1772758 returned 403 | verified |
| 4 | M. Curtis-Maury, F. Blagojevic, C. D. Antonopoulos, D. S. Nikolopoulos. *Prediction-Based Power-Performance Adaptation of Multithreaded Scientific Codes.* IEEE Transactions on Parallel and Distributed Systems 19(10):1396-1410, 2008. doi:10.1109/TPDS.2007.70804 | Paper PDF, author group copy (14 pages) | https://faculty.e-ce.uth.gr/cda/docs/tpds08.pdf : title and 4 authors on page 1. Journal, volume, issue, pages, year, DOI from the Crossref record (https://api.crossref.org/works?query.bibliographic=Prediction-Based+Power-Performance+Adaptation...). IEEE page (doi.org redirect to https://ieeexplore.ieee.org/document/4407685/) not readable (bot challenge) | verified (venue via Crossref) |
| 5 | K. K. Pusukuri, R. Gupta, L. N. Bhuyan. *Thread Reinforcer: Dynamically Determining Number of Threads via OS Level Monitoring.* 2011 IEEE International Symposium on Workload Characterization (IISWC), pp. 116-125. doi:10.1109/IISWC.2011.6114208 | Paper PDF, author copy (10 pages) | http://alumni.cs.ucr.edu/~kishore/papers/iiswc62-pusukuri.pdf : title and 3 authors (UC Riverside) on page 1. Venue, date (Nov 2011), pages, DOI from https://api.crossref.org/works/10.1109/iiswc.2011.6114208 . IEEE page https://ieeexplore.ieee.org/document/6114208 not readable | verified (venue via Crossref) |
| 6 | B. Yang, L. He, N. Ling, Z. Yan, G. Xing, X. Shuai, X. Ren, X. Jiang. *EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge.* ACM SenSys 2023. arXiv:2311.10986 | arXiv abstract page | https://arxiv.org/abs/2311.10986 : title, 8 authors, v1 18 Nov 2023, comment "Accepted to the 21th ACM Conference on Embedded Networked Sensor Systems (SenSys 2023)" | verified |

## Facts quoted in the report from the papers

| Ref | Fact in final_report.html section 17 | Where checked |
|---|---|---|
| 5 | 24-core server; thread count giving maximum speedup ranged from 16 to 63 threads across PARSEC programs (max speedups 3.6x to 21.9x) | page 1 abstract of the author PDF above |
| 4 | Multivariate regression on hardware event counters; phase-aware run-time throttling of concurrency | page 1 abstract of the author PDF above |
| 6 | Switches between on-device and cloud foundation model at run time based on data uncertainty and network conditions | arXiv abstract |
| 1 | 1.1B parameters, pretrained on about 1 trillion tokens | arXiv abstract |

## Not cited

| Reference | Why |
|---|---|
| C. Hu, W. Bao, D. Wang, F. Liu. *Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge.* IEEE INFOCOM 2019, pp. 1423-1431, doi:10.1109/INFOCOM.2019.8737614 | Metadata confirmed on Crossref (https://api.crossref.org/works/10.1109/infocom.2019.8737614) and the PolyU research portal (https://research.polyu.edu.hk/en/publications/dynamic-adaptive-dnn-surgery-for-inference-acceleration-on-the-ed/), but the paper itself could not be opened: the author PDF https://web.comp.polyu.edu.hk/csdwang/Publication/INFOCOM19-Edge.pdf returns 404 and IEEE Xplore is blocked. Left out under the "only cite papers you have opened" rule. |
| Kang et al., Neurosurgeon (ASPLOS 2017) | Not found by OpenAlex title search; not opened. |
| Holm (1979), sequentially rejective multiple test procedure | Not opened (JSTOR); the method is described in the report without a citation. |
| Pushkar 2020 (IEEE Micro); Zhang/Xiao/Shen 2022 (IEEE TVT) | From an earlier draft; not found. Not reused. |

Earlier drafts' attributions that were wrong and are not reused: "Hu et al." is the INFOCOM 2019 DNN-surgery paper above;
EdgeFM is arXiv 2311.10986 by Yang et al. (not the authors given earlier); Curtis-Maury 2008 is the TPDS paper above;
Pusukuri 2011 is Thread Reinforcer with Gupta and Bhuyan.
