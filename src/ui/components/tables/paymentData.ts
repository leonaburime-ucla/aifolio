


export type Payment = {
  id: string;
  amount: number;
  status: "pending" | "processing" | "success" | "failed";
  email: string;
  customer: string;
  method: "card" | "ach" | "wire" | "crypto";
  region: string;
  riskScore: number;
};

export const payments: Payment[] = [
  {
    id: "m5gr84i9-0",
    amount: 316,
    status: "success",
    email: "ken99.com",
    customer: "Ken Jacobs",
    method: "card",
    region: "US-West",
    riskScore: 0.12
  },
  {
    id: "3u1reuv4-1",
    amount: 245,
    status: "success",
    email: "Abe45.com",
    customer: "Abigail Peters",
    method: "ach",
    region: "US-East",
    riskScore: 0.33
  },
  {
    id: "derv1ws0-2",
    amount: 843,
    status: "processing",
    email: "Monserrat44.com",
    customer: "Monserrat Rollins",
    method: "wire",
    region: "EU-Central",
    riskScore: 0.6
  },
  {
    id: "5kma53ae-3",
    amount: 883,
    status: "success",
    email: "Silas22.com",
    customer: "Silas Meyer",
    method: "card",
    region: "US-West",
    riskScore: 0.28
  },
  {
    id: "bhqecj4p-4",
    amount: 733,
    status: "failed",
    email: "carmella.com",
    customer: "Carmella Burke",
    method: "crypto",
    region: "APAC",
    riskScore: 0.75
  },
  {
    id: "m5gr84i9-5",
    amount: 331,
    status: "success",
    email: "ken99.com",
    customer: "Ken Jacobs",
    method: "card",
    region: "US-West",
    riskScore: 0.17
  },
  {
    id: "3u1reuv4-6",
    amount: 260,
    status: "success",
    email: "Abe45.com",
    customer: "Abigail Peters",
    method: "ach",
    region: "US-East",
    riskScore: 0.38
  },
  {
    id: "derv1ws0-7",
    amount: 858,
    status: "processing",
    email: "Monserrat44.com",
    customer: "Monserrat Rollins",
    method: "wire",
    region: "EU-Central",
    riskScore: 0.65
  },
  {
    id: "5kma53ae-8",
    amount: 898,
    status: "success",
    email: "Silas22.com",
    customer: "Silas Meyer",
    method: "card",
    region: "US-West",
    riskScore: 0.33
  },
  {
    id: "bhqecj4p-9",
    amount: 748,
    status: "failed",
    email: "carmella.com",
    customer: "Carmella Burke",
    method: "crypto",
    region: "APAC",
    riskScore: 0.8
  },
  {
    id: "m5gr84i9-10",
    amount: 346,
    status: "success",
    email: "ken99.com",
    customer: "Ken Jacobs",
    method: "card",
    region: "US-West",
    riskScore: 0.22
  },
  {
    id: "3u1reuv4-11",
    amount: 275,
    status: "success",
    email: "Abe45.com",
    customer: "Abigail Peters",
    method: "ach",
    region: "US-East",
    riskScore: 0.43
  },
  {
    id: "derv1ws0-12",
    amount: 873,
    status: "processing",
    email: "Monserrat44.com",
    customer: "Monserrat Rollins",
    method: "wire",
    region: "EU-Central",
    riskScore: 0.7
  },
  {
    id: "5kma53ae-13",
    amount: 913,
    status: "success",
    email: "Silas22.com",
    customer: "Silas Meyer",
    method: "card",
    region: "US-West",
    riskScore: 0.38
  },
  {
    id: "bhqecj4p-14",
    amount: 763,
    status: "failed",
    email: "carmella.com",
    customer: "Carmella Burke",
    method: "crypto",
    region: "APAC",
    riskScore: 0.85
  },
  {
    id: "m5gr84i9-15",
    amount: 361,
    status: "success",
    email: "ken99.com",
    customer: "Ken Jacobs",
    method: "card",
    region: "US-West",
    riskScore: 0.27
  },
  {
    id: "3u1reuv4-16",
    amount: 290,
    status: "success",
    email: "Abe45.com",
    customer: "Abigail Peters",
    method: "ach",
    region: "US-East",
    riskScore: 0.48
  },
  {
    id: "derv1ws0-17",
    amount: 888,
    status: "processing",
    email: "Monserrat44.com",
    customer: "Monserrat Rollins",
    method: "wire",
    region: "EU-Central",
    riskScore: 0.75
  },
  {
    id: "5kma53ae-18",
    amount: 928,
    status: "success",
    email: "Silas22.com",
    customer: "Silas Meyer",
    method: "card",
    region: "US-West",
    riskScore: 0.43
  },
  {
    id: "bhqecj4p-19",
    amount: 778,
    status: "failed",
    email: "carmella.com",
    customer: "Carmella Burke",
    method: "crypto",
    region: "APAC",
    riskScore: 0.9
  },
  {
    id: "m5gr84i9-20",
    amount: 376,
    status: "success",
    email: "ken99.com",
    customer: "Ken Jacobs",
    method: "card",
    region: "US-West",
    riskScore: 0.32
  },
  {
    id: "3u1reuv4-21",
    amount: 305,
    status: "success",
    email: "Abe45.com",
    customer: "Abigail Peters",
    method: "ach",
    region: "US-East",
    riskScore: 0.53
  },
  {
    id: "derv1ws0-22",
    amount: 903,
    status: "processing",
    email: "Monserrat44.com",
    customer: "Monserrat Rollins",
    method: "wire",
    region: "EU-Central",
    riskScore: 0.8
  },
  {
    id: "5kma53ae-23",
    amount: 943,
    status: "success",
    email: "Silas22.com",
    customer: "Silas Meyer",
    method: "card",
    region: "US-West",
    riskScore: 0.48
  },
  {
    id: "bhqecj4p-24",
    amount: 793,
    status: "failed",
    email: "carmella.com",
    customer: "Carmella Burke",
    method: "crypto",
    region: "APAC",
    riskScore: 0.95
  }
];
