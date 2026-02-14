/**
 * Mock transistor data for testing
 */

export const mockTransistor1 = {
  metadata: {
    name: 'CREE_C3M0060065J',
    type: 'SiC-MOSFET',
    manufacturer: 'CREE',
    housing_type: 'TO-247-3',
    author: 'Test Author',
    comment: 'Test transistor 1'
  },
  electrical: {
    v_abs_max: 650,
    i_abs_max: 120,
    i_cont: 60,
    t_j_max: 175
  },
  thermal: {
    r_th_cs: 0.24,
    r_th_total: 0.5,
    housing_area: 0.0012,
    cooling_area: 0.0008
  },
  switch: {
    channel_data: [
      {
        v_g: 15,
        dataset_type: 'graph_i_e',
        t_j: 25,
        graph_v_i: [[0, 0.5, 1.0, 1.5], [0, 10, 20, 30]]
      }
    ],
    e_on_data: [
      {
        dataset_type: 'graph_e_i',
        v_supply: 400,
        v_g: 15,
        t_j: 25,
        r_g: 10,
        graph_i_e: [[0, 10, 20, 30], [0, 0.5, 1.0, 1.5]]
      }
    ],
    e_off_data: [],
    thermal_foster: {
      r_th_vector: [0.1, 0.14],
      c_th_vector: [0.001, 0.01],
      tau_vector: [0.0001, 0.0014]
    }
  },
  diode: {
    channel_data: [
      {
        v_g: -2,
        dataset_type: 'graph_i_e',
        t_j: 25,
        graph_v_i: [[0, 0.7, 1.4], [0, 10, 20]]
      }
    ],
    e_rr_data: [],
    thermal_foster: {
      r_th_vector: [0.15, 0.2],
      c_th_vector: [0.002, 0.015],
      tau_vector: [0.0003, 0.003]
    }
  },
  c_oss: [
    {
      dataset_type: 'graph_c_v',
      v_g: 0,
      graph_v_c: [[0, 100, 200, 300], [1000, 500, 300, 200]]
    }
  ],
  c_iss: [],
  c_rss: []
}

export const mockTransistor2 = {
  metadata: {
    name: 'Infineon_FF300R12KE3',
    type: 'IGBT',
    manufacturer: 'Infineon',
    housing_type: 'PrimePACK3',
    author: 'Test Author',
    comment: 'Test transistor 2'
  },
  electrical: {
    v_abs_max: 1200,
    i_abs_max: 600,
    i_cont: 300,
    t_j_max: 150
  },
  thermal: {
    r_th_cs: 0.05,
    r_th_total: 0.15,
    housing_area: 0.005,
    cooling_area: 0.004
  },
  switch: {
    channel_data: [],
    e_on_data: [],
    e_off_data: [],
    thermal_foster: {
      r_th_vector: [0.05, 0.1],
      c_th_vector: [0.005, 0.05],
      tau_vector: [0.00025, 0.005]
    }
  },
  diode: {
    channel_data: [],
    e_rr_data: [],
    thermal_foster: {
      r_th_vector: [0.08, 0.12],
      c_th_vector: [0.004, 0.04],
      tau_vector: [0.00032, 0.0048]
    }
  },
  c_oss: [],
  c_iss: [],
  c_rss: []
}

export const mockTransistor3 = {
  metadata: {
    name: 'GaNSystems_GS66506T',
    type: 'GaN-Transistor',
    manufacturer: 'GaN Systems',
    housing_type: 'GaNPX',
    author: 'Test Author',
    comment: 'Test transistor 3'
  },
  electrical: {
    v_abs_max: 650,
    i_abs_max: 60,
    i_cont: 30,
    t_j_max: 150
  },
  thermal: {
    r_th_cs: 0.5,
    r_th_total: 1.0,
    housing_area: 0.0003,
    cooling_area: 0.0002
  },
  switch: {
    channel_data: [],
    e_on_data: [],
    e_off_data: [],
    thermal_foster: {
      r_th_vector: [0.3, 0.2],
      c_th_vector: [0.0001, 0.001],
      tau_vector: [0.00003, 0.0002]
    }
  },
  diode: {
    channel_data: [],
    e_rr_data: [],
    thermal_foster: {
      r_th_vector: [0.4, 0.3],
      c_th_vector: [0.00015, 0.0015],
      tau_vector: [0.00006, 0.00045]
    }
  },
  c_oss: [],
  c_iss: [],
  c_rss: []
}

export const mockTransistors = [
  mockTransistor1,
  mockTransistor2,
  mockTransistor3
]

export const emptyTransistor = {
  metadata: {
    name: '',
    type: '',
    manufacturer: '',
    housing_type: '',
    author: '',
    comment: ''
  },
  electrical: {
    v_abs_max: 0,
    i_abs_max: 0,
    i_cont: 0,
    t_j_max: 0
  },
  thermal: {
    r_th_cs: 0,
    r_th_total: 0,
    housing_area: 0,
    cooling_area: 0
  },
  switch: {
    channel_data: [],
    e_on_data: [],
    e_off_data: [],
    thermal_foster: {
      r_th_vector: [],
      c_th_vector: [],
      tau_vector: []
    }
  },
  diode: {
    channel_data: [],
    e_rr_data: [],
    thermal_foster: {
      r_th_vector: [],
      c_th_vector: [],
      tau_vector: []
    }
  },
  c_oss: [],
  c_iss: [],
  c_rss: []
}
